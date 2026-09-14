#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture fixed teacher-forced M32 gate/up inputs with the Mach runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, assets, sha, write_json


def configure(mxfp6):
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("EXL3_", "VLLM_MACH_", "VLLM_EXL3_", "B12X_"))}
    profile = ROOT / "profiles/vllm-0.29.0/qwen38-checkpoint-fp16-ssm.env"
    raw = subprocess.check_output(["bash", "-c", 'source "$1"; env -0', "mach-calibration", str(profile)], env=env)
    os.environ.clear()
    os.environ.update(dict(x.decode().split("=", 1) for x in raw.split(b"\0") if x))
    os.environ.update(VLLM_MACH_MXFP6_CHECKPOINT=str(mxfp6.resolve()),
                      VLLM_USE_V2_MODEL_RUNNER="1", K4O_QUALITY_ARM="m32")
    # Installed only in this calibration subprocess, never the normal launcher.
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")


def install(worker, offsets):
    import torch
    import teacher_decode
    import chunked_teacher
    from vllm_mach.exl3 import mxfp6_hybrid, fused_allreduce
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import fused_allreduce_gemma_rms_norm
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_decode.install(worker)
    if offsets[-1] == 1000:
        chunked_teacher.install(worker)
    # Preserve the fused BF16 AR/RMSNorm path while exposing BF16 GU inputs.
    fused_allreduce.fused_allreduce_gemma_rms_norm_mxfp8 = fused_allreduce_gemma_rms_norm
    names = {f"language_model.model.layers.{i}.mlp.gate_up_proj" for i in assets.MASK}
    found = {m.prefix: m for m in worker.model_runner.model.modules() if getattr(m, "prefix", "") in names}
    if set(found) != names:
        raise ValueError("Expected all 48 selected gate/up modules")
    ids = {}
    for name, layer in found.items():
        state = mxfp6_hybrid.state_for_rows(layer, 32)
        if state is None or state.merged_weight is None:
            raise ValueError("Calibration requires original MXFP6 gate/up weights")
        ids[id(state.merged_weight)] = name
    original = mxfp6_hybrid.apply_weight
    calls = {name: [] for name in names}

    def observed(x, weight):
        result = original(x, weight)
        history = worker._k4o_teacher_sampler._k4o_observations
        name = ids.get(id(weight))
        if history and name is not None:
            offset = history[-1]["offset"] + 1
            if offset in offsets:
                if tuple(x.shape) != (32, 5120) or x.dtype != torch.bfloat16:
                    raise ValueError("Calibration must observe BF16 physical M32")
                previous = calls[name]
                if previous and offset <= previous[-1]["decode_offset"]:
                    raise ValueError("Duplicate or out-of-order capture")
                previous.append({"decode_offset": offset, "sample_ids": list(history[-1]["ids"]),
                                 "x": x.detach().clone()})
        return result

    mxfp6_hybrid.apply_weight = observed
    worker._mach_capture = (calls, offsets, original)


def save(worker, directory):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_mach.exl3 import mxfp6_hybrid
    calls, offsets, original = worker._mach_capture
    rank = int(get_tensor_model_parallel_rank())
    try:
        torch.cuda.synchronize()
        for records in calls.values():
            if [r["decode_offset"] for r in records] != offsets:
                raise ValueError("Incomplete M32 capture")
            for row in records:
                if not torch.isfinite(row["x"]).all():
                    raise ValueError("Nonfinite captured input")
                row["x"] = row["x"].cpu()
        path = Path(directory) / f"gu_inputs_rank{rank}.pt"
        with path.open("xb") as stream:
            torch.save({"rank": rank, "layers": {k: {"calls": v} for k, v in calls.items()}}, stream)
        return {"rank": rank, "file": path.name, "sha256": sha(path)}
    finally:
        mxfp6_hybrid.apply_weight = original


def llm_arguments(args, manifest):
    long = manifest["prompt_tokens"] == 3000
    return dict(model=str(args.model), tokenizer=str(args.tokenizer), quantization="exl3",
                dtype="bfloat16", tensor_parallel_size=2, max_num_seqs=32,
                max_model_len=4000 if long else 200, max_num_batched_tokens=4096 if long else 6112,
                long_prefill_token_threshold=128 if long else 0,
                kv_cache_memory_bytes=8218214400, num_gpu_blocks_override=627,
                kv_cache_dtype="auto", mamba_ssm_cache_dtype="float16",
                enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True,
                attention_backend="TRITON_ATTN", language_model_only=True, seed=20260907,
                logprobs_mode="raw_logprobs", disable_log_stats=True,
                compilation_config={"mode": "NONE", "cudagraph_mode": "NONE"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "mxfp6", "tokenizer", "manifest", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    configure(args.mxfp6)
    if importlib.metadata.version("vllm") != "0.29.0":
        raise RuntimeError("Calibration capture requires the Mach vLLM 0.29.0 image")
    import torch
    import teacher_decode
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    if len(samples) != 32 or len({s["id"] for s in samples}) != 32:
        raise ValueError("Expected 32 fixed calibration samples")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    llm = LLM(**llm_arguments(args, manifest))
    llm.collective_rpc(install, args=(manifest["decode_offsets"],))
    prompts, params, sequences = [], [], []
    for sample in samples:
        sequence = [sample["prompt_token_ids"][-1]] + sample["target_token_ids"]
        sequences.append(sequence)
        prompts.append(TokensPrompt(prompt_token_ids=sample["prompt_token_ids"][:-1]))
        params.append(SamplingParams(temperature=0, max_tokens=len(sequence), ignore_eos=True,
                      logprobs=1, detokenize=False, extra_args={teacher_decode.KEY: {
                          "id": sample["id"], "forced_ids": sequence, "pad_id": sequence[-1]}}))
    core = llm.llm_engine.engine_core
    core.call_utility("pause_scheduler", "keep", False)
    if not core.call_utility("is_scheduler_paused"):
        raise RuntimeError("Calibration requires a paused scheduler before enqueue")
    internal = llm.enqueue(prompts, params, use_tqdm=False)
    states = llm.llm_engine.output_processor.request_states
    external = [str(states[str(i)].external_req_id) for i in internal]
    core.call_utility("resume_scheduler")
    results = {str(x.request_id): x for x in llm.wait_for_completion(use_tqdm=False)}
    if len(external) != 32 or set(results) != set(external):
        raise ValueError("Incomplete calibration requests")
    for key, sequence in zip(external, sequences, strict=True):
        if list(results[key].outputs[0].token_ids) != sequence:
            raise ValueError("Teacher-forced sequence mismatch")
    records = llm.collective_rpc(save, args=(str(args.output.resolve()),))
    if manifest["prompt_tokens"] == 3000:
        import chunked_teacher
        llm.collective_rpc(chunked_teacher.observations)
    write_json(args.output / "COMPLETE.json", {"manifest_sha256": sha(args.manifest),
               "records": records, "runtime": {n: importlib.metadata.version(n) for n in (
                   "vllm", "torch", "exllamav3", "flashinfer-python", "vllm-mach")}})
    # This subprocess exits before the next capture/model load starts.


if __name__ == "__main__":
    main()
