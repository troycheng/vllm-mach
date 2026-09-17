#!/usr/bin/env python3
"""Eager TP2 diagnostics for Qwen35 fused AR/Norm and the optional NVFP4 head.

Uses real model hidden states. Deliberately performs shadow communication and
full BF16 head GEMMs, so this must never be used as a throughput benchmark.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def install_ar_probe(worker):
    import torch
    import vllm.model_executor.models.qwen3_5 as qwen
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _can_use_flashinfer,
    )

    original = qwen.fused_allreduce_gemma_rms_norm
    worker._ar_probe = []
    seen = set()

    def observed(hidden, residual, norm):
        rows = hidden.shape[0]
        if rows in seen:
            return original(hidden, residual, norm)
        seen.add(rows)
        assert _can_use_flashinfer(hidden, 2)[0], "fused transport was not available"
        reference = norm(
            tensor_model_parallel_all_reduce(hidden.clone()), residual.clone()
        )
        actual = original(hidden, residual, norm)
        errors = []
        for result, expected in zip(actual, reference, strict=True):
            error = float(
                (result.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-9)
            )
            assert torch.isfinite(result).all()
            assert error < 0.02, error
            errors.append(error)
        worker._ar_probe.append(
            {"rows": rows, "relative_l2": errors, "flashinfer_fused": True}
        )
        return actual

    qwen.fused_allreduce_gemma_rms_norm = observed


def ar_stats(worker):
    model = worker.model_runner.model.language_model.model
    layers = list(model.layers)
    assert model.use_fused_ar_gemma_norm
    assert all(layer.mlp.experts.moe_config.skip_final_all_reduce for layer in layers)
    assert all(
        hasattr(layer.mlp.experts, "_mxfp6_sm120_qwen35_small_batch_state")
        for layer in layers
    )
    assert worker._ar_probe
    return {
        "rank": worker.rank,
        "fused_layers": len(layers),
        "native_moe_schedules_retained": True,
        "samples": worker._ar_probe,
    }


class WorkerExtension:
    def mach_install_probes(self):
        from fidelity_native_mxfp6 import install_head_probe

        install_head_probe(self)
        install_ar_probe(self)

    def mach_head_stats(self):
        from fidelity_native_mxfp6 import head_stats

        return head_stats(self)

    def mach_ar_stats(self):
        return ar_stats(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    from vllm_mach.mxfp6.serve import profile_environment

    os.environ.update(
        profile_environment(
            SimpleNamespace(
                model=args.model,
                fp16_ssm=False,
                lossless_prefill=False,
                owner_prefill=False,
                nvfp4_lm_head=True,
                verify_prefill=False,
            )
        )
    )
    from vllm import LLM, SamplingParams

    llm = LLM(
        worker_extension_cls="verify_qwen35_optimizations.WorkerExtension",
        model=str(args.model),
        tensor_parallel_size=2,
        dtype="bfloat16",
        quantization="quark",
        max_model_len=4096,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        kv_cache_memory_bytes=8589934592,
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_backend="TRITON_ATTN",
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={"mode": "NONE"},
        generation_config="vllm",
    )
    llm.collective_rpc("mach_install_probes")
    prompts = [
        "请解释为什么天空是蓝色的。",
        "Write a Python function to sort a list.",
        "Calculate 17 times 19 and explain the steps.",
        "Compare renewable energy sources in three sentences.",
        "把这句话翻译成英文：我们明天去北京。",
        "What is the capital of France?",
        "Explain matrix multiplication with a small example.",
        "Describe the water cycle.",
    ]
    tokenizer = llm.get_tokenizer()
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    for batch in (1, 4, 16, 32):
        llm.generate(
            [rendered[i % len(rendered)] for i in range(batch)],
            SamplingParams(
                temperature=0,
                max_tokens=args.output_tokens,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
    heads = llm.collective_rpc("mach_head_stats")
    result = {
        "model": args.model.name,
        "batches": [1, 4, 16, 32],
        "output_tokens": args.output_tokens,
        "head": heads,
        "allreduce_norm": llm.collective_rpc("mach_ar_stats"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for rank in heads:
        print({key: value for key, value in rank.items() if key != "calls"})
        assert rank["global_argmax_mismatches"] == 0
        assert rank["global_winner_candidate_misses_on_rank"] == 0
    print("AR/Norm and NVFP4 head diagnostics passed")


if __name__ == "__main__":
    main()
