#!/usr/bin/env python3
"""Frozen256 physical-M4/M32 raw decode diagnostic; never used for throughput."""

import argparse
import importlib.metadata
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quantization"))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def manifest(path):
    samples = json.loads(Path(path).read_text())["samples"]
    assert len(samples) == len({s["id"] for s in samples}) == 256
    assert sum(len(s["target_token_ids"]) for s in samples) == 10479
    for s in samples:
        assert s["full_token_ids"] == s["prompt_token_ids"] + s["target_token_ids"]
    return samples


def gdn_stats(worker):
    from vllm_mach.mxfp6.gdn_decode import stats

    return {"rank": worker.rank, **stats()}


def install_teacher(worker):
    import teacher_decode

    return teacher_decode.install(worker, native=True)


def install_head_probe(worker):
    """Compare the actual compact greedy head against full BF16 on identical states."""
    import types

    import torch
    from vllm.distributed import tensor_model_parallel_all_gather

    from vllm_mach.mxfp6.argmax_triton import reduce_global_argmax_triton

    target = getattr(
        worker.model_runner.model, "language_model", worker.model_runner.model
    )
    processor = target.logits_processor
    original = processor.get_top_tokens
    worker._head_stats = []

    def observed(this, head, hidden, embedding_bias=None):
        sampled = original(head, hidden, embedding_bias)
        state = head._hybrid_nvfp4_lm_head_state
        start = head.shard_indices.org_vocab_start_index
        end = head.shard_indices.org_vocab_end_index
        full = this._apply_head(head, hidden, embedding_bias)
        full[:, end - start :] = -float("inf")
        values, indices = full.max(-1)
        pairs = torch.stack([values.float(), (indices + start).float()], -1)
        reference = reduce_global_argmax_triton(
            tensor_model_parallel_all_gather(pairs, dim=-1), tp_size=head.tp_size
        )
        coarse = state.coarse_logits(hidden, embedding_bias)
        coarse[:, end - start :] = -float("inf")
        candidates = state.select_candidates(coarse)
        refined = state.refine_logits(hidden, head.weight, candidates, embedding_bias)
        selected = full.gather(1, candidates.long())
        missing = ((reference >= start) & (reference < end)) & ~(
            candidates == (reference - start)[:, None]
        ).any(-1)
        local_values, local_indices = full.topk(20, dim=-1)
        global_values = tensor_model_parallel_all_gather(local_values, dim=-1)
        global_indices = tensor_model_parallel_all_gather(local_indices + start, dim=-1)
        top20 = global_indices.gather(1, global_values.topk(20, dim=-1).indices)
        owned = (top20 >= start) & (top20 < end)
        retained = (top20[:, :, None] == (candidates + start)[:, None, :]).any(-1)
        missing20 = owned & ~retained
        all_missing = tensor_model_parallel_all_gather(
            missing20.any(-1).int()[:, None], dim=-1
        ).any(-1)
        worker._head_stats.append(
            dict(
                rows=hidden.shape[0],
                global_argmax_mismatches=int((sampled != reference).sum()),
                global_winner_candidate_misses_on_rank=int(missing.sum()),
                global_top20_candidate_misses_on_rank=int(missing20.sum()),
                rows_missing_any_global_top20=int(all_missing.sum()),
                selected_logit_mismatches=int((refined != selected).sum()),
                max_selected_logit_error=float(
                    (refined.float() - selected.float()).abs().max()
                ),
            )
        )
        return sampled

    processor.get_top_tokens = types.MethodType(observed, processor)


def head_stats(worker):
    from vllm.distributed import get_tensor_model_parallel_rank

    rows = worker._head_stats
    assert rows, "NVFP4 head was not exercised"
    return dict(
        rank=get_tensor_model_parallel_rank(),
        calls=rows,
        **{
            k: sum(r[k] for r in rows)
            for k in rows[0]
            if k != "max_selected_logit_error"
        },
        max_selected_logit_error=max(r["max_selected_logit_error"] for r in rows),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm",
        choices=[
            "bf16",
            "default",
            "gdn",
            "persistent",
            "full",
            "full_ba",
            "full_gdn",
            "fp8",
            "nvfp4",
        ],
        required=True,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head-only", action="store_true")
    p.add_argument("--physical-rows", type=int, choices=[4, 32], default=32)
    p.add_argument("--skip-head-probe", action="store_true")
    p.add_argument(
        "--no-stock-compile",
        action="store_true",
        help="Diagnostic ablation only; not the stock serving configuration",
    )
    a = p.parse_args()
    samples = manifest(a.manifest)
    from vllm_mach.mxfp6.serve import profile_environment

    full = a.arm in ("full", "full_ba", "full_gdn")
    if a.head_only and not full:
        p.error("--head-only requires --arm full or full_ba")
    if a.head_only and (a.skip_head_probe or a.physical_rows != 32):
        p.error("--head-only requires M32 and cannot use --skip-head-probe")
    flags = profile_environment(
        argparse.Namespace(
            fp16_ssm=full,
            gdn_persistent=a.arm in ("persistent", "gdn", "full_gdn"),
            gdn_ba_overlap=a.arm in ("full_ba", "gdn", "full_gdn"),
            lossless_prefill=full,
            owner_prefill=full,
            nvfp4_lm_head=full,
            verify_prefill=False,
        )
    )
    if a.arm in ("bf16", "fp8", "nvfp4"):
        flags.update(VLLM_QWEN3_5_FUSED_AR_NORM="0", VLLM_VOCAB_PARALLEL_GREEDY="0")
    if a.arm in ("bf16", "fp8", "nvfp4"):
        flags["VLLM_PLUGINS"] = ""
    os.environ.update(flags)
    os.environ["K4O_QUALITY_ARM"] = "m32"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["PYTHONPATH"] = (
        str(ROOT / "tools/quantization") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    import teacher_decode
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=False)
    eager = a.arm == "bf16"
    args = dict(
        model=a.model,
        tokenizer=a.tokenizer,
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_num_seqs=32,
        max_model_len=768,
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=3489660928,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=eager,
        attention_backend="TRITON_ATTN",
        language_model_only=True,
        seed=20260907,
        logprobs_mode="raw_logprobs",
        disable_log_stats=True,
        mamba_ssm_cache_dtype="float16" if full else "float32",
        compilation_config=dict(
            mode="NONE",
            cudagraph_mode="NONE" if eager else "FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32],
        ),
    )
    if a.arm in ("default", "gdn", "persistent", "full", "full_ba", "full_gdn"):
        args["quantization"] = "quark"
    if a.arm in ("fp8", "nvfp4") and not a.no_stock_compile:
        # Keep stock compilation enabled, as in the stock throughput baselines.
        args["compilation_config"] = dict(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32]
        )
    if eager:
        args["cpu_offload_gb"] = 4
        os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"
    library_hash = None
    if a.arm not in ("bf16", "fp8", "nvfp4"):
        import mxfp6
        library_hash = hashlib.sha256(Path(mxfp6.load_library()).read_bytes()).hexdigest()
    write(
        a.output / "contract.json",
        dict(
            arm=a.arm,
            manifest=str(a.manifest),
            physical_rows=a.physical_rows,
            extension_library_sha256=library_hash,
            fused_gdn_quant=os.environ.get("VLLM_MACH_FUSED_GDN_QUANT", "auto"),
            fused_swiglu_quant=os.environ.get("VLLM_MACH_FUSED_SWIGLU_QUANT", "auto"),
            llm_args=args,
            environment=flags,
            packages={
                n: importlib.metadata.version(n)
                for n in [
                    "vllm",
                    "torch",
                    "flashinfer-python",
                    "vllm-mach",
                    "mxfp6-sm120",
                ]
            },
        ),
    )
    llm = LLM(**args)
    if not a.head_only:
        write(a.output / "hook.json", llm.collective_rpc(install_teacher))
    rows_per_batch = a.physical_rows
    cohorts = len(samples) // rows_per_batch
    if not a.head_only and rows_per_batch != 32:
        llm.collective_rpc(teacher_decode.set_physical_rows, args=(rows_per_batch,))
    records = []
    for index in range(0 if a.head_only else cohorts + 1):
        batch = samples[
            (index % cohorts) * rows_per_batch : (index % cohorts + 1) * rows_per_batch
        ]
        max_gold = max(s["target_tokens"] for s in batch)
        assert sum(len(s["prompt_token_ids"]) - 1 for s in batch) <= 8192
        prompts, params, sequences = [], [], []
        for s in batch:
            seq = [s["prompt_token_ids"][-1]] + s["target_token_ids"]
            seq += [seq[-1]] * (max_gold - s["target_tokens"])
            assert len(s["prompt_token_ids"]) - 1 + len(seq) <= 768
            prompts.append(TokensPrompt(prompt_token_ids=s["prompt_token_ids"][:-1]))
            sequences.append(seq)
            params.append(
                SamplingParams(
                    temperature=0,
                    max_tokens=len(seq),
                    ignore_eos=True,
                    logprobs=1,
                    detokenize=False,
                    extra_args={
                        teacher_decode.KEY: dict(
                            id=s["id"], forced_ids=seq, pad_id=seq[-1]
                        )
                    },
                )
            )
        core = llm.llm_engine.engine_core
        core.call_utility("pause_scheduler", "keep", False)
        assert core.call_utility("is_scheduler_paused")
        internal = llm.enqueue(prompts, params, use_tqdm=False)
        states = llm.llm_engine.output_processor.request_states
        external = [str(states[str(i)].external_req_id) for i in internal]
        core.call_utility("resume_scheduler")
        results = {
            str(r.request_id): r for r in llm.wait_for_completion(use_tqdm=False)
        }
        assert len(results) == rows_per_batch and set(results) == set(external)
        rows = []
        for s, seq, key in zip(batch, sequences, external, strict=True):
            output = results[key].outputs[0]
            assert list(output.token_ids) == seq and output.finish_reason == "length"
            values = [
                float(output.logprobs[j + 1][t].logprob)
                for j, t in enumerate(s["target_token_ids"])
            ]
            assert all(math.isfinite(v) for v in values)
            rows.append(dict(id=s["id"], domain=s["domain"], gold_logprobs=values))
        calls = llm.collective_rpc(teacher_decode.observations)
        for rank in calls:
            scored = [c for c in rank["calls"] if 1 <= c["offset"] <= max_gold]
            assert len(scored) == max_gold
            assert {c["offset"] for c in scored} == set(range(1, max_gold + 1))
            assert all(
                c["num_tokens"] == c["num_reqs"] == rows_per_batch
                and set(c["ids"]) == {s["id"] for s in batch}
                for c in scored
            )
        write(
            a.output / f"batch{index:02}.json", dict(records=rows, runtime_calls=calls)
        )
        if index < cohorts:
            records.extend(rows)
        else:
            delta = [
                abs(x - y)
                for r, s in zip(records[:rows_per_batch], rows, strict=True)
                for x, y in zip(r["gold_logprobs"], s["gold_logprobs"], strict=True)
            ]
            write(
                a.output / "repeat.json",
                dict(max_abs=max(delta), mean_abs=sum(delta) / len(delta)),
            )
        print("FIDELITY_BATCH", a.arm, index, flush=True)
    if not a.head_only:
        write(a.output / "records.json", records)
    if full and not a.skip_head_probe:
        llm.collective_rpc(install_head_probe)
        for index in range(8):
            batch = samples[index * 32 : (index + 1) * 32]
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=s["prompt_token_ids"]) for s in batch],
                SamplingParams(
                    temperature=0, max_tokens=48, ignore_eos=True, detokenize=False
                ),
                use_tqdm=False,
            )
            assert len(outputs) == 32 and all(
                len(o.outputs[0].token_ids) == 48 for o in outputs
            )
        write(a.output / "head.json", llm.collective_rpc(head_stats))
    if a.arm in ("default", "gdn", "persistent", "full", "full_ba", "full_gdn"):
        write(a.output / "gdn.json", llm.collective_rpc(gdn_stats))
    write(
        a.output / "COMPLETE.json",
        dict(
            arm=a.arm,
            queries=256,
            **(
                {"requested_tokens_per_query": 48, "mode": "head-only"}
                if a.head_only
                else {"target_tokens": 10479}
            ),
        ),
    )


if __name__ == "__main__":
    main()
