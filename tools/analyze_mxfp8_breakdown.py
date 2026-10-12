"""Partition MXFP8 decode CUDA traces into disjoint compute categories."""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import importlib.metadata
import json
import re
import statistics
import subprocess
from pathlib import Path

CATEGORIES = ("GEMM", "full_attention", "GDN", "other")
LAYER_TYPES = ["full_attention" if i % 4 == 3 else "GDN" for i in range(32)]


def gemm(name):
    return (
        any(part in name.lower() for part in ("gemm", "gemv", "splitkreduce"))
        and "quantize" not in name.lower()
    )


def mx_gemm(name):
    return gemm(name) and not any(s in name.lower() for s in ("fp4", "f4e2m1")) and (
        "mxfp8" in name.lower()
        or "MainloopSm120TmaWarpSpecializedBlockScaled" in name
        or "blockscaled_gemm_sm120_b12xDenseGemmKernel" in name
    )


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total, end = 0, float("-inf")
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def classify(kernel, norms):
    name = kernel["name"]
    index = bisect.bisect_right([n["ts"] for n in norms], kernel["ts"]) - 1
    attention_region = 0 <= index < 64 and index % 2 == 0
    owner = LAYER_TYPES[index // 2] if attention_region else "other"
    if index >= 64 and "indexed_bf16_dot" in name:
        return "GEMM", "BF16_lm_head_refinement"
    if gemm(name):
        if mx_gemm(name):
            return "GEMM", "MXFP8_projections"
        if index >= 64:
            return "GEMM", "NVFP4_lm_head" if any(s in name.lower() for s in ("fp4", "f4e2m1")) else "BF16_lm_head"
        return "GEMM", "BF16_BA_projection"
    if "quantize_mx_kernel" in name:
        return "other", "activation_quantization"
    if name == "_silu_quant":
        return "other", "fused_SwiGLU_quantization"
    if name == "_norm_quant":
        return "other", "fused_residual_RMSNorm_quantization"
    if name == "_norm":
        return "other", "residual_RMSNorm"
    if owner == "full_attention":
        if any(
            x in name for x in ("attention", "flash_fwd", "fmha", "reduce_segments")
        ):
            return owner, "attention_core"
        return owner, "QK_norm_RoPE_KV_cache_gate_layout"
    if owner == "GDN":
        if "gdn_fused_decode_kernel" in name:
            return owner, "persistent_BA_conv_recurrence"
        if "delta_rule" in name:
            return owner, "recurrence"
        if "conv1d" in name:
            return owner, "causal_conv"
        if "layer_norm" in name:
            return owner, "gated_RMSNorm"
        if any(
            part in name
            for part in (
                "chunk_",
                "recompute_w_u",
                "inverse_kernel",
                "_fused_post_conv",
                "prepare_wy",
                "solve_tril",
            )
        ):
            return owner, "prefill_chunk_delta_rule"
        return owner, "copies_zero_layout"
    return "other", "embedding_SwiGLU_sampling_scheduler_misc"


def decode_profile(path, batch, phase="decode"):
    with gzip.open(path) as trace_file:
        events = json.load(trace_file)["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
    )
    step_ends = [
        e["ts"] + e["dur"]
        for e in kernels
        if e["name"] == "_scatter_num_accepted_kernel"
    ]
    steps = [[] for _ in step_ends]
    for kernel in kernels:
        idx = bisect.bisect_left(step_ends, kernel["ts"])
        if idx >= len(steps):
            raise ValueError(f"Unassigned trailing kernel: {kernel['name']}")
        steps[idx].append(kernel)
    annotations = collections.defaultdict(list)
    for event in events:
        if (
            event.get("cat") == "gpu_user_annotation"
            and "execute_context" in event["name"]
        ):
            annotations[(event["name"], event["args"]["External id"])].append(event)
    assert len(annotations) == len(steps), (path, len(annotations), len(steps))
    if phase == "decode":
        launch_path = path.parent / 'launch.json'
        expected_steps = (json.loads(launch_path.read_text()).get('contract', {}).get('profile_steps', 40)
                          if launch_path.exists() else 40)
        assert len(steps) == expected_steps, (path, len(steps), expected_steps)
        assert {name for name, _ in annotations} == {
            f"execute_context_0(0)_generation_{batch}({batch})"
        }, (path, {name for name, _ in annotations})
    else:
        assert all("_generation_0(0)" in name for name, _ in annotations), path
        tokens = sum(
            int(re.search(r"execute_context_\d+\((\d+)\)", name).group(1))
            for name, _ in annotations
        )
        assert tokens == batch * 3000, (path, tokens)
    totals, details = collections.Counter(), collections.Counter()
    counts, names, projections = (
        collections.Counter(),
        {},
        collections.defaultdict(list),
    )
    samples = []
    for step in steps:
        norms = [e for e in step if e["name"] in ("_norm", "_norm_quant")]
        assert len(norms) == 65, (path, len(norms))
        mx = [e for e in step if mx_gemm(e["name"])]
        assert len(mx) == 128, (path, len(mx))
        for idx, kernel in enumerate(mx):
            layer, stage = divmod(idx, 4)
            kind = LAYER_TYPES[layer]
            label = (
                f"{kind}_input"
                if stage == 0
                else f"{kind}_output"
                if stage == 1
                else "MLP_gate_up"
                if stage == 2
                else "MLP_down"
            )
            projections[label].append(kernel["dur"] / 1000)
        groups = collections.Counter()
        for kernel in step:
            category, subcategory = classify(kernel, norms)
            groups[category] += kernel["dur"] / 1000
            details[f"{category}/{subcategory}"] += kernel["dur"] / 1000
            counts[f"{category}/{subcategory}"] += 1
            entry = names.setdefault(
                (kernel["name"], category, subcategory),
                {"sum_ms": 0, "calls": 0, "category": category, "detail": subcategory},
            )
            entry["sum_ms"] += kernel["dur"] / 1000
            entry["calls"] += 1
        totals.update(groups)
        samples.append(
            {
                "groups_ms": dict(groups),
                "kernel_sum_ms": sum(groups.values()),
                "gpu_busy_union_ms": union_us(step) / 1000,
                "gpu_span_ms": (
                    max(e["ts"] + e["dur"] for e in step) - min(e["ts"] for e in step)
                )
                / 1000,
                "kernels": len(step),
            }
        )
    n = len(steps)
    forward = [
        (max(e["ts"] + e["dur"] for e in group) - min(e["ts"] for e in group)) / 1000
        for group in annotations.values()
    ]
    result = {
        "trace": str(path.resolve()),
        "batch": batch,
        "steps": n,
        "phase": phase,
        "groups_ms": {c: totals[c] / n for c in CATEGORIES},
        "details_ms": {k: v / n for k, v in details.items()},
        "calls_per_step": {k: v / n for k, v in counts.items()},
        "kernel_sum_ms": sum(totals.values()) / n,
        "kernels_per_step": len(kernels) / n,
        "model_forward_median_ms": statistics.median(forward),
        "model_forward_samples_ms": forward,
        "gpu_span_median_ms": statistics.median(s["gpu_span_ms"] for s in samples),
        "gpu_busy_union_mean_ms": statistics.fmean(
            s["gpu_busy_union_ms"] for s in samples
        ),
        "samples": samples,
        "projection_shapes": {
            label: {
                "sum_ms_per_step": sum(values) / n,
                "calls_per_step": len(values) / n,
                "mean_us_per_call": statistics.fmean(values) * 1000,
            }
            for label, values in projections.items()
        },
        "kernels": [
            {
                "name": k[0],
                "mean_ms_per_step": v["sum_ms"] / n,
                "calls_per_step": v["calls"] / n,
                "category": v["category"],
                "detail": v["detail"],
            }
            for k, v in sorted(names.items(), key=lambda kv: -kv[1]["sum_ms"])
        ],
        "cuda_allocation_calls": dict(
            collections.Counter(
                e["name"]
                for e in events
                if "Malloc" in e.get("name", "") or "Free" in e.get("name", "")
            )
        ),
        "gpu_copy_memset_ms_per_step": {
            category: sum(e.get("dur", 0) for e in events if e.get("cat") == category)
            / n
            / 1000
            for category in ("gpu_memcpy", "gpu_memset")
        },
    }
    serving_path = path.parent / f"bs{batch}-serving.json"
    if serving_path.exists():
        measurements = json.loads(serving_path.read_text())
        result["serving"] = {
            "repeats": len(measurements),
            "mean_tpot_ms": statistics.fmean(r["mean_tpot_ms"] for r in measurements),
            "tpot_samples_ms": [r["mean_tpot_ms"] for r in measurements],
            "mean_ttft_ms": statistics.fmean(r["mean_ttft_ms"] for r in measurements),
            "output_throughput_tokens_s": statistics.fmean(
                r["output_throughput_tokens_s"] for r in measurements
            ),
            "all_completed": all(
                row["success"] for r in measurements for row in r["requests"]
            ),
            "steady_stream_median_itl_ms": statistics.median(
                interval * 1000
                for r in measurements
                for row in r["requests"]
                for interval in row["itl_s"][80:160]
            ),
            "steady_stream_median_itl_samples_ms": [
                statistics.median(
                    interval * 1000
                    for row in r["requests"]
                    for interval in row["itl_s"][80:160]
                )
                for r in measurements
            ],
            "steady_stream_note": "Median of visible-text SSE intervals 80:160 per request; some SSE events group tokens, so this is supporting evidence, not exact per-token accounting",
        }
    if phase == "prefill":
        result["total_groups_ms"] = dict(totals)
        result["total_kernel_sum_ms"] = sum(totals.values())
        result["total_gpu_busy_union_ms"] = union_us(kernels) / 1000
        result["total_gpu_span_ms"] = (
            max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
        ) / 1000
        result["input_tokens"] = tokens
        result["annotations"] = dict(
            collections.Counter(name for name, _ in annotations)
        )
    return result


def output_checks(root, profiles):
    result = {}
    for batch in [1, 16, 32]:
        base = json.loads((root / "default" / f"bs{batch}-serving.json").read_text())
        for arm in sorted({name.rsplit("_bs", 1)[0] for name in profiles}):
            path = root / arm / f"bs{batch}-serving.json"
            if not path.exists():
                continue
            samples = json.loads(path.read_text())
            matches = sum(
                u["response_sha256"] == v["response_sha256"]
                for a, b in zip(base, samples)
                for u, v in zip(a["requests"], b["requests"])
            )
            repeats = sum(
                samples[0]["requests"][i]["response_sha256"]
                == samples[r]["requests"][i]["response_sha256"]
                for r in range(1, len(samples))
                for i in range(batch)
            )
            result[f"{arm}_bs{batch}"] = {
                "text_matches_vs_default": matches,
                "text_comparisons_vs_default": min(len(base), len(samples)) * batch,
                "text_matches_repeated_runs": repeats,
                "text_comparisons_repeated_runs": (len(samples) - 1) * batch,
                "note": f"{samples[0]['output_tokens']}-token greedy generation text hash smoke check; does not establish logit equivalence or model accuracy",
            }
    return result


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    baseline = [result["profiles"][f"default_bs{b}"] for b in [1, 16, 32]]
    bottom = np.zeros(3)
    colors = ["#4263eb", "#20a68e", "#e8a53e", "#9e77c2"]
    for category, color in zip(CATEGORIES, colors):
        values = np.array([r["groups_ms"][category] for r in baseline])
        axes[0].bar(
            ["BS1", "BS16", "BS32"],
            values,
            bottom=bottom,
            label=category.replace("_", " "),
            color=color,
        )
        for x, value, base in zip(range(3), values, bottom):
            if value > 0.4:
                axes[0].text(
                    x,
                    base + value / 2,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if category == "GEMM" else "black",
                )
        bottom += values
    axes[0].set_ylabel("Sum of GPU kernel durations / decode step (ms)")
    axes[0].set_title("Default command: native MXFP8, BF16 head")
    axes[0].legend(loc="upper left")
    for arm, style in [
        ("default", "o-"),
        ("b12x", "s--"),
        ("gdn", "^-"),
        ("gdn_flash_attn", "d--"),
        ("pdl", "s--"),
        ("norm_quant", "^-"),
        ("norm_quant_pdl", "d--"),
        ("nvfp4_head", "x-"),
        ("nvfp4_head_pdl", "x-"),
    ]:
        if all(f"{arm}_bs{b}" in result["profiles"] for b in [1, 16, 32]):
            values = [
                result["profiles"][f"{arm}_bs{b}"]["gpu_span_median_ms"]
                for b in [1, 16, 32]
            ]
            axes[1].plot(range(3), values, style, label=arm)
    axes[1].set_xticks(range(3), ["BS1", "BS16", "BS32"])
    axes[1].set_ylabel("Median GPU span, including head + sampling (ms)")
    contract = result["contract"]
    axes[1].set_title(
        f"GPU {result['cuda_visible_devices']} / TP1 / "
        f"{contract['input_tokens']} in, {contract['serving_generation_tokens']} out\n"
        f"{contract['decode_steps_per_trace']} fixed decode steps per trace"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = {
        "date": "2026-09-30",
        "model": "/data1/models/Qwen3.5-4B-MXFP8",
        "tensor_parallel_size": 1,
        "cuda_visible_devices": "2",
        "gpu": "NVIDIA GeForce RTX 5090 / SM120",
        "reference_repo": "/data/luyufan/vllm-shpgy",
        "reference_commit": "1d0591245c3be6bd702a226600972722ecee023b",
        "contract": {
            "input_tokens": 3000,
            "decode_steps_per_trace": 40,
            "profiler_delay_iterations": 80,
            "generation_tokens_per_profile_request": 192,
            "serving_repeats": 3,
            "serving_generation_tokens": 256,
            "prompts": "docs/data/serving-prompts.json: first BS frozen ShareGPT prompts repeated to 3000 tokens",
            "compilation": "{}: default VLLM_COMPILE / FULL_AND_PIECEWISE",
            "category_definition": "GEMM includes all MXFP8 projections, BF16 BA, BF16/NVFP4 head and BF16 candidate refinement; attention/GDN exclude these projections; fused persistent BA belongs to GDN",
            "attribution": "65 RMSNorm or fused norm/quant kernels delimit the 32 layers; 3 GDN / 1 full attention repeated 8 times; generic copies/zeros assigned by layer region",
            "other": "common residual/RMSNorm, activation quantization, fused SwiGLU+quant, embedding, sampling and scheduling",
            "timing": "category values sum CUDA kernel durations; overlap counts twice; GPU span is first-to-last GPU kernel, and busy union removes overlap; profiler CPU/idle gaps can inflate span",
            "serving": "HTTP TPOT = (last response-first visible text)/(completion tokens-1), includes mixed prefill during initial batch fill and later tail, plus frontend overhead; not fixed-BS CUDA step time",
        },
        "profiles": {},
        "prefill_profiles": {},
    }
    launches = [json.loads(p.read_text()) for p in sorted(args.input.glob('*/launch.json'))]
    devices = {launch['environment']['CUDA_VISIBLE_DEVICES'] for launch in launches}
    if len(devices) == 1:
        result['cuda_visible_devices'] = devices.pop()
    contracts = [launch['contract'] for launch in launches if 'contract' in launch]
    if contracts:
        assert all(c == contracts[0] for c in contracts), "Mixed benchmark contracts"
        contract = contracts[0]
        result['contract'].update({
            'generation_tokens_per_profile_request': contract['output_tokens'],
            'serving_generation_tokens': contract['output_tokens'],
            'serving_repeats': contract['repeats'],
            'warmup_tokens': contract['warmup_tokens'],
            'profiler_delay_iterations': contract['profile_delay'],
            'decode_steps_per_trace': contract['profile_steps'],
        })
    result['launches'] = launches
    for path in sorted(args.input.glob("*/bs*.pt.trace.json.gz")):
        batch = int(re.search(r"bs(\d+)", path.name).group(1))
        result["profiles"][f"{path.parent.name}_bs{batch}"] = decode_profile(
            path, batch
        )
    for path in sorted(args.input.glob("prefill/*/bs*.pt.trace.json.gz")):
        batch = int(re.search(r"bs(\d+)", path.name).group(1))
        result["prefill_profiles"][f"{path.parent.name}_bs{batch}"] = decode_profile(
            path, batch, "prefill"
        )
    result["output_smoke_checks"] = output_checks(args.input, result["profiles"])
    result["versions"] = {
        n: importlib.metadata.version(n)
        for n in ["torch", "vllm", "flashinfer-python", "mxfp6-sm120", "vllm-mach"]
    }
    result["gpu_snapshot"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv",
        ],
        text=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w") as f:
        fields = [
            "arm",
            "batch",
            *CATEGORIES,
            "kernel_sum_ms",
            "gpu_span_median_ms",
            "model_forward_median_ms",
            "HTTP_TPOT_ms",
            "steady_SSE_ITL_median_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, row in result["profiles"].items():
            writer.writerow(
                dict(
                    arm=label.rsplit("_bs", 1)[0],
                    batch=row["batch"],
                    **row["groups_ms"],
                    kernel_sum_ms=row["kernel_sum_ms"],
                    gpu_span_median_ms=row["gpu_span_median_ms"],
                    model_forward_median_ms=row["model_forward_median_ms"],
                    HTTP_TPOT_ms=row.get("serving", {}).get("mean_tpot_ms"),
                    steady_SSE_ITL_median_ms=row.get("serving", {}).get(
                        "steady_stream_median_itl_ms"
                    ),
                )
            )
            print(
                label,
                {k: round(v, 4) for k, v in row["groups_ms"].items()},
                "sum",
                round(row["kernel_sum_ms"], 4),
                "GPU span",
                round(row["gpu_span_median_ms"], 4),
            )
    for label, row in result["prefill_profiles"].items():
        print(
            "PREFILL",
            label,
            {k: round(v, 3) for k, v in row["total_groups_ms"].items()},
            "sum",
            round(row["total_kernel_sum_ms"], 3),
        )
    with args.output.with_name(args.output.stem + "-components.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "arm", "batch", "component", "ms", "calls", "basis"])
        for phase, profiles in [
            ("decode", result["profiles"]),
            ("prefill", result["prefill_profiles"]),
        ]:
            for label, row in profiles.items():
                factor = 1 if phase == "decode" else row["steps"]
                for component, value in row["details_ms"].items():
                    writer.writerow(
                        [
                            phase,
                            label.rsplit("_bs", 1)[0],
                            row["batch"],
                            component,
                            value * factor,
                            row["calls_per_step"][component] * factor,
                            "per decode step"
                            if phase == "decode"
                            else "entire prefill batch",
                        ]
                    )
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(result, args.plot)


if __name__ == "__main__":
    main()
