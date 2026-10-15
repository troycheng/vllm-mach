#!/usr/bin/env python3
"""Compare actual stochastic decode distributions on identical hidden states.

Diagnostic only: never enable this observer in a throughput service. Frozen
prompts are read from the project's fidelity manifest; model data stay local.
"""

import argparse
import json
import os
from pathlib import Path


def install_probe(worker):
    import torch
    from vllm.distributed import tensor_model_parallel_all_gather

    from vllm_mach.mxfp6.sampling import SamplingHead

    original = SamplingHead._get_local_topk
    target = getattr(
        worker.model_runner.model, "language_model", worker.model_runner.model
    )
    state = getattr(target.lm_head, "_hybrid_nvfp4_lm_head_state", None)
    if state is None:
        state = getattr(worker.model_runner, "_mach_sampling_sidecar", None)
    assert state is not None and state.backend == "b12x", "NVFP4 head was not prepared"
    worker._sampling_precision = []

    def observed(self, head, hidden, **kwargs):
        assert hidden.shape[0] == worker._sampling_precision_rows, (
            "Physical batch changed"
        )
        state = self._get_hybrid_lm_head_state(head)
        assert state is not None and state.can_use(
            hidden,
            bf16_weight=head.weight,
            active_vocab_size=head.shard_indices.org_vocab_end_index
            - head.shard_indices.org_vocab_start_index,
            top_k=kwargs["top_k"],
        ), "Precision probe did not exercise NVFP4 search"
        result = original(self, head, hidden, **kwargs)
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

        k = kwargs["top_k"]
        values, ids, start, mapping = result
        ids = self._local_indices_to_global(
            ids, vocab_start=start, shard_token_ids=mapping
        )
        # Independent full-vocabulary reference, including threshold ties at k.
        reference = head.quant_method.apply(
            head, hidden, bias=kwargs.get("embedding_bias")
        ).float()
        # The 4B profile's original head is K2048, not a full BF16 projection.
        # Measure the incremental change against that serving baseline as well.
        baseline = (
            self.processor._apply_head(
                head, hidden, kwargs.get("embedding_bias")
            ).float()
            if os.environ.get("VLLM_MACH_PROFILE") == "qwen35-4b-mxfp8-champion-v1"
            else reference.clone()
        )
        if self.soft_cap is not None:
            reference = torch.tanh(reference / self.soft_cap) * self.soft_cap
            baseline = torch.tanh(baseline / self.soft_cap) * self.soft_cap
        reference *= self.scale
        baseline *= self.scale
        if kwargs.get("presence_penalties") is not None:
            self._apply_sharded_presence_penalty_from_counts(
                reference,
                kwargs["presence_penalties"],
                kwargs["output_token_counts"],
                kwargs["presence_request_indices"],
                shard_indices=head.shard_indices,
            )
            self._apply_sharded_presence_penalty_from_counts(
                baseline,
                kwargs["presence_penalties"],
                kwargs["output_token_counts"],
                kwargs["presence_request_indices"],
                shard_indices=head.shard_indices,
            )
        reference /= kwargs["temperature"]
        baseline /= kwargs["temperature"]
        active = (
            head.shard_indices.org_vocab_end_index
            - head.shard_indices.org_vocab_start_index
        )
        self._mask_invalid_shard_logits(reference, None, active)
        self._mask_invalid_shard_logits(baseline, None, active)
        if head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            reference = tensor_model_parallel_all_gather(reference, dim=-1)
            baseline = tensor_model_parallel_all_gather(baseline, dim=-1)
        assert reference.shape[1] == self.org_vocab_size, (
            "Probe requires contiguous unpadded TP vocabulary"
        )
        bi = reference.topk(k, dim=-1).indices
        p = worker._sampling_precision_top_p
        av, ai = self._select_from_compact_topk_pairs(
            torch.stack((values, ids.float()), -1), k, p
        )
        filtered = apply_top_k_top_p_pytorch(
            reference,
            torch.full((hidden.shape[0],), k, dtype=torch.int32, device=hidden.device),
            torch.full(
                (hidden.shape[0],), p, dtype=torch.float32, device=hidden.device
            ),
        )
        ap, bp = av.softmax(-1), filtered.softmax(-1)
        bp_on_a = bp.gather(1, ai.long())
        retained = torch.zeros_like(bp, dtype=torch.bool).scatter_(1, ai.long(), True)
        missing_mass = bp.masked_fill(retained, 0).sum(-1)
        tv = 0.5 * ((ap - bp_on_a).abs().sum(-1) + missing_mass)
        baseline_filtered = apply_top_k_top_p_pytorch(
            baseline,
            torch.full((hidden.shape[0],), k, dtype=torch.int32, device=hidden.device),
            torch.full(
                (hidden.shape[0],), p, dtype=torch.float32, device=hidden.device
            ),
        ).softmax(-1)
        baseline_missing = baseline_filtered.masked_fill(retained, 0).sum(-1)
        baseline_tv = 0.5 * (
            (ap - baseline_filtered.gather(1, ai.long())).abs().sum(-1)
            + baseline_missing
        )
        same = ai[:, :, None] == bi[:, None, :]
        # Report probabilities rather than arbitrary stochastic token equality.
        worker._sampling_precision.append(
            {
                "rows": hidden.shape[0],
                "top_k": k,
                "total_variation_sum": float(tv.sum()),
                "total_variation_max": float(tv.max()),
                "missing_reference_mass_sum": float(missing_mass.sum()),
                "rows_missing_reference_support": int((missing_mass > 0).sum()),
                "global_topk_overlap": int(same.any(2).sum()),
                "baseline_total_variation_sum": float(baseline_tv.sum()),
                "baseline_total_variation_max": float(baseline_tv.max()),
            }
        )
        return result

    SamplingHead._get_local_topk = observed
    return {"rank": worker.rank, "installed": True}


def begin_probe(worker, top_p, physical_rows):
    worker._sampling_precision = []
    worker._sampling_precision_top_p = top_p
    worker._sampling_precision_rows = physical_rows


def generate_fixed_cohort(llm, prompts, params):
    """Queue every request before scheduling, matching native fidelity runs."""
    core = llm.llm_engine.engine_core
    core.call_utility("pause_scheduler", "keep", False)
    try:
        assert core.call_utility("is_scheduler_paused")
        llm.enqueue(prompts, params, use_tqdm=False)
    finally:
        core.call_utility("resume_scheduler")
    return llm.wait_for_completion(use_tqdm=False)


def select_sampling(worker, enabled):
    import os

    os.environ["VLLM_MACH_LM_HEAD_SAMPLING"] = str(int(enabled))


def fallback_signature(outputs):
    return [
        {
            "tokens": list(o.outputs[0].token_ids),
            "scores": None
            if o.outputs[0].logprobs is None
            else [
                {str(k): float(v.logprob) for k, v in row.items()}
                for row in o.outputs[0].logprobs
            ],
        }
        for o in outputs
    ]


def read_probe(worker):
    rows = worker._sampling_precision
    count = sum(r["rows"] for r in rows)
    return {
        "rank": worker.rank,
        "calls": len(rows),
        "rows": count,
        "mean_total_variation": sum(r["total_variation_sum"] for r in rows)
        / max(count, 1),
        "max_total_variation": max((r["total_variation_max"] for r in rows), default=0),
        "mean_missing_reference_mass": sum(
            r["missing_reference_mass_sum"] for r in rows
        )
        / max(count, 1),
        "rows_missing_reference_support": sum(
            r["rows_missing_reference_support"] for r in rows
        ),
        "topk_overlap": sum(r["global_topk_overlap"] for r in rows),
        "mean_baseline_total_variation": sum(
            r["baseline_total_variation_sum"] for r in rows
        )
        / max(count, 1),
        "max_baseline_total_variation": max(
            (r["baseline_total_variation_max"] for r in rows), default=0
        ),
        "dispatch": getattr(worker.model_runner, "_mach_sampling_stats", {}),
    }


class SamplingDiagnostics:
    """Named local worker RPCs; no pickle fallback is required."""

    def mach_install_sampling_probe(self):
        return install_probe(self)

    def mach_begin_sampling_probe(self, top_p, physical_rows):
        return begin_probe(self, top_p, physical_rows)

    def mach_select_sampling(self, enabled):
        return select_sampling(self, enabled)

    def mach_read_sampling_probe(self):
        return read_probe(self)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/data/fidelity-samples.json",
    )

    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[4, 32])
    p.add_argument("--prompts", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument(
        "--profile",
        choices=("native", "nvfp4", "mxfp8-2b", "mxfp8-4b"),
        default="native",
    )
    a = p.parse_args()
    if a.profile in ("native", "nvfp4"):
        os.environ.update(
            VLLM_PLUGINS="mach",
            VLLM_USE_V2_MODEL_RUNNER="1",
            VLLM_MACH_LM_HEAD_SAMPLING="1",
            VLLM_VOCAB_PARALLEL_GREEDY="1",
            VLLM_HYBRID_NVFP4_LM_HEAD="1",
            VLLM_HYBRID_NVFP4_LM_HEAD_BACKEND="b12x",
            VLLM_HYBRID_NVFP4_LM_HEAD_CANDIDATES="128",
            VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS="32",
            VLLM_SM120_OWNER_PREFILL="0",
            VLLM_SM120_LOSSLESS_PREFILL="0",
            VLLM_MACH_FUSED_AR_QUANT="0",
            VLLM_QWEN3_5_FUSED_AR_NORM="1",
            VLLM_MACH_GDN_PERSISTENT="1",
            VLLM_MACH_GDN_BA_OVERLAP="1",
            VLLM_QWEN3_5_FP16_SSM="0",
            VLLM_ALLREDUCE_USE_FLASHINFER="0",
            VLLM_FLASHINFER_ALLREDUCE_BACKEND="trtllm",
            VLLM_USE_BREAKABLE_CUDAGRAPH="0",
            MXFP6_AUTOTUNE="off",
        )
        if a.profile == "nvfp4":
            os.environ.update(
                VLLM_QWEN3_5_FUSED_AR_NORM="0",
                VLLM_MACH_GDN_PERSISTENT="0",
                VLLM_MACH_GDN_BA_OVERLAP="0",
                VLLM_MACH_MOE_PROJECTION_TUNING="0",
            )
    else:
        if len(a.rows) != 1:
            raise ValueError("Run each MXFP8 quality geometry in its own process")
        if a.profile == "mxfp8-2b":
            from vllm_mach.mxfp8.two_b.model import validate_model
            from vllm_mach.mxfp8.two_b.profile import (
                check_environment,
                check_runtime_sources,
                quality_contract,
            )
            from vllm_mach.mxfp8.two_b.serve import configure_environment

            configure_environment(
                a.output.parent / (a.output.stem + "-runtime"),
                quality_rows=max(a.rows),
                attention_backend="TRITON_ATTN",
                hybrid_head=True,
            )
        else:
            from vllm_mach.mxfp8.prepare_model import validate_model
            from vllm_mach.mxfp8.profile import (
                check_environment,
                check_runtime_sources,
                quality_contract,
            )
            from vllm_mach.mxfp8.serve import configure_environment

            configure_environment(
                a.output.parent / (a.output.stem + "-runtime"),
                quality_rows=max(a.rows),
                hybrid_head=True,
            )
        check_environment()
        check_runtime_sources()
        validate_model(a.model, verify_weights=True)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    samples = json.loads(a.manifest.read_text())["samples"][: a.prompts]
    kwargs = {
        "quantization": "quark",
        "dtype": "bfloat16",
        "tensor_parallel_size": a.tp,
        "max_num_seqs": max(a.rows),
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "kv_cache_memory_bytes": 4294967296,
        "enable_prefix_caching": False,
        "language_model_only": True,
        "attention_backend": "TRITON_ATTN",
        "mamba_ssm_cache_dtype": "float32",
        "kv_cache_dtype": "bfloat16",
        "generation_config": "vllm",
        "compilation_config": {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32],
        },
    }
    if a.profile == "nvfp4":
        kwargs["quantization"] = "compressed-tensors"
    elif a.profile != "native":
        kwargs = quality_contract(max(a.rows)).copy()
        if a.profile == "mxfp8-2b":
            kwargs.update(
                dtype="bfloat16",
                tensor_parallel_size=1,
                quantization="compressed-tensors",
                attention_backend="TRITON_ATTN",
                language_model_only=True,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                mamba_ssm_cache_dtype="float32",
                kv_cache_dtype="bfloat16",
                seed=20260907,
                logprobs_mode="raw_logprobs",
                disable_log_stats=True,
                kernel_config={"linear_backend": "flashinfer_cutlass"},
            )
    llm = LLM(
        model=a.model,
        worker_extension_cls="verify_lm_head_sampling.SamplingDiagnostics",
        **kwargs,
    )
    fallback_checks = []
    for params in [
        SamplingParams(
            temperature=0.7,
            top_k=20,
            top_p=0.95,
            seed=20261015,
            max_tokens=16,
            ignore_eos=True,
        ),
        SamplingParams(temperature=0.0, logprobs=5, max_tokens=16, ignore_eos=True),
    ]:
        outputs = []
        for enabled in [False, True]:
            llm.collective_rpc("mach_select_sampling", args=(enabled,))
            outputs.append(
                fallback_signature(
                    generate_fixed_cohort(
                        llm,
                        [
                            TokensPrompt(prompt_token_ids=s["prompt_token_ids"])
                            for s in samples[:4]
                        ],
                        params,
                    )
                )
            )
        assert outputs[0] == outputs[1], "Seeded/logprob fallback changed"
        fallback_checks.append(
            {
                "explicit_seed": params.seed is not None,
                "logprobs": params.logprobs,
                "identical": True,
            }
        )
    llm.collective_rpc("mach_install_sampling_probe")
    results = []
    for rows in a.rows:
        for k, presence in [(20, 0.0), (20, 1.5), (64, 0.0)]:
            llm.collective_rpc("mach_begin_sampling_probe", args=(0.95, rows))
            count = 0
            for offset in range(0, len(samples), rows):
                batch = samples[offset : offset + rows]
                if len(batch) < rows:
                    raise ValueError("Prompt count must be divisible by physical rows")
                outputs = generate_fixed_cohort(
                    llm,
                    [
                        TokensPrompt(prompt_token_ids=s["prompt_token_ids"])
                        for s in batch
                    ],
                    SamplingParams(
                        temperature=0.7,
                        top_k=k,
                        top_p=0.95,
                        presence_penalty=presence,
                        max_tokens=a.output_tokens,
                        ignore_eos=True,
                    ),
                )
                assert len(outputs) == rows and all(
                    len(o.outputs[0].token_ids) == a.output_tokens for o in outputs
                )
                count += len(outputs)
            result = {
                "physical_rows": rows,
                "top_k": k,
                "top_p": 0.95,
                "temperature": 0.7,
                "presence_penalty": presence,
                "completed": count,
                "ranks": llm.collective_rpc("mach_read_sampling_probe"),
            }
            assert all(
                r["rows"] == count * (a.output_tokens - 1) for r in result["ranks"]
            ), result
            results.append(result)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(
                json.dumps(
                    {
                        "profile": a.profile,
                        "fallback_checks": fallback_checks,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
