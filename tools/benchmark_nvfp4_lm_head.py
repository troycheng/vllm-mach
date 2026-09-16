#!/usr/bin/env python3
"""Compare the actual TP-local LM-head GEMM APIs on identical NVFP4 operands."""

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

import torch
from safetensors import safe_open


def measure(fn, repeats, trials):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            _result = fn()
    # Warm the captured workload long enough to avoid idle-clock ramp effects.
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def load_kineto(path):
    spec = importlib.util.spec_from_file_location("reference_gemm_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.bench_kineto


def measure_kineto(fn, bench, repeats, trials):
    # Discover all device kernels for one call, including repeated helper kernels.
    # The reference helper returns mean time per matching kernel launch; multiply
    # by its per-call multiplicity before summing a multi-kernel operation.
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    counts = Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    # Internal reduction-workspace clears can share the same profiler name as
    # the reference's 8 GB L2 flush. Exclude memory operations rather than ever
    # matching that flush as part of the operator's kernel-only timing.
    excluded = {n: c for n, c in counts.items() if n.startswith(("Memset", "Memcpy"))}
    counts = Counter({n: c for n, c in counts.items() if n not in excluded})
    assert counts, counts
    # Match the displayed names, whose width is capped at 100 by bench_kineto.
    prefixes = Counter()
    for name, count in counts.items():
        prefixes[name[:90]] += count
    samples, kernel_samples = [], []
    for _ in range(trials):
        times = bench(
            fn,
            tuple(prefixes),
            num_tests=repeats,
            suppress_kineto_output=True,
            flush_l2=True,
            with_multiple_kernels=True,
        )
        assert len(times) == len(prefixes) and all(t > 0 for t in times), (
            prefixes,
            times,
        )
        samples.append(
            sum(t * count for t, count in zip(times, prefixes.values(), strict=True))
            * 1e6
        )
        kernel_samples.append([t * 1e6 for t in times])
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "kernels_per_call": dict(counts),
        "excluded_device_ops_per_call": excluded,
        "matched_prefixes": dict(prefixes),
        "per_kernel_mean_us": kernel_samples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument("--rank", type=int, choices=[0, 1], default=0)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timer", choices=["graph", "kineto"], default="kineto")
    p.add_argument("--gemm-bench", type=Path, default=Path("/data/lxy/gemm_bench.py"))
    a = p.parse_args()
    if a.timer == "kineto":
        bench = load_kineto(a.gemm_bench)

        def timer(fn):
            return measure_kineto(fn, bench, a.repeats, a.trials)
    else:

        def timer(fn):
            return measure(fn, a.repeats, a.trials)

    from b12x.gemm.blockscaled import mm_nvfp4
    from flashinfer import mm_fp4
    from vllm.utils.flashinfer import (
        autotune_with_torch_cuda_delay,
    )
    from vllm.utils.flashinfer import (
        flashinfer_nvfp4_quantize_128x4 as quantize,
    )
    from vllm.utils.flashinfer import (
        flashinfer_scaled_fp4_mm as wrapped,
    )

    from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import _global_scale

    torch.manual_seed(20260915)
    index = json.loads((a.model / "model.safetensors.index.json").read_text())
    with safe_open(
        a.model / index["weight_map"]["lm_head.weight"], framework="pt"
    ) as f:
        head = f.get_slice("lm_head.weight")
        vocab, k = head.get_shape()
        assert vocab % 256 == 0
        n = vocab // 2
        weight = head[a.rank * n : (a.rank + 1) * n].to(
            device="cuda", dtype=torch.bfloat16
        )
    scale = _global_scale(weight)
    wq, ws = quantize(weight, scale)
    del weight
    results = []
    for m in a.rows:
        hidden = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        hs = _global_scale(hidden)
        hq, hsf = quantize(hidden, hs)
        alpha = torch.reciprocal(hs * scale)

        def run(name, q=hq, sf=hsf, factor=alpha, m=m):
            if name == "vllm_flashinfer_b12x":
                return wrapped(q, wq, sf, ws, factor, torch.bfloat16, "b12x")
            if name == "standalone_b12x":
                return mm_nvfp4(q, sf, wq, ws, factor, expected_m=m)
            backend = "cutlass" if name == "flashinfer_cutlass" else "b12x"
            return mm_fp4(q, wq.t(), sf, ws.t(), factor, backend=backend)

        def coarse(name, hidden=hidden, run=run):
            current_scale = _global_scale(hidden)
            q, sf = quantize(hidden, current_scale)
            return run(name, q, sf, torch.reciprocal(current_scale * scale))

        names = [
            "vllm_flashinfer_b12x",
            "flashinfer_b12x",
            "standalone_b12x",
            "flashinfer_cutlass",
        ]
        with autotune_with_torch_cuda_delay(tune_mode=True):
            run("flashinfer_b12x")
            run("flashinfer_cutlass")
        reference = run("vllm_flashinfer_b12x").clone()
        for name in names:
            print(f"MEASURE M={m} backend={name} timer={a.timer}", flush=True)
            output = run(name)
            torch.cuda.synchronize()
            assert output.shape == (m, n) and torch.isfinite(output).all()
            delta = (output.float() - reference.float()).abs()
            row = {
                "m": m,
                "n": n,
                "k": k,
                "backend": name,
                "max_abs_vs_current": float(delta.max()),
                "mean_abs_vs_current": float(delta.mean()),
                "top1_agreement": float(
                    (output.argmax(-1) == reference.argmax(-1)).float().mean()
                ),
                "gemm": timer(lambda name=name: run(name)),
                "quantize_and_gemm": timer(lambda name=name: coarse(name)),
            }
            results.append(row)
            print(
                json.dumps(
                    {
                        "m": m,
                        "backend": name,
                        "gemm_us": row["gemm"]["median_us"],
                        "quantize_and_gemm_us": row["quantize_and_gemm"]["median_us"],
                    }
                ),
                flush=True,
            )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "packages": {
                        x: metadata.version(x)
                        for x in ["torch", "vllm", "flashinfer-python", "b12x"]
                    },
                    "model": str(a.model),
                    "tp_rank": a.rank,
                    "seed": 20260915,
                    "protocol": "Fixed real head shard, random BF16 activations; identical quantization; FlashInfer tuned, standalone expected_m heuristic; excludes top-k/refinement/TP communication",
                    "timer": a.timer,
                    "reference_timer": str(a.gemm_bench)
                    if a.timer == "kineto"
                    else None,
                    "flush_l2_bytes": 8_000_000_000 if a.timer == "kineto" else 0,
                    "reduction": "Median of trial means; sums per-kernel mean times times launch multiplicities; excludes L2 flush, internal Memset/Memcpy, CPU overhead and launch gaps"
                    if a.timer == "kineto"
                    else "Median of graph replay averages",
                    "repeats": a.repeats,
                    "trials": a.trials,
                    "graph_warmup_replays": 20 if a.timer == "graph" else 0,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
