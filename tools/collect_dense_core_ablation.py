#!/usr/bin/env python3
"""Summarize the cumulative reconstruction of the Dense base Mach profile."""

import argparse
import json
from pathlib import Path

from collect_dense_ablation import collect, markdown

GROUPS = {
    "runtime": ["fp8", "core_graph", "core_sampler", "core_ar_norm", "core_gdn", "core_ba"],
    "producer": ["fp8", "core_ba", "core_swiglu", "core_gdn_quant", "core_ar_quant"],
}
COMPARISONS = {
    "runtime": [
        ("Minimum MXFP6 vs user FP8 service (graphs enabled)", "fp8", "core_graph"),
        ("Compact greedy sampler", "core_graph", "core_sampler"),
        ("AR/residual/RMSNorm fusion", "core_sampler", "core_ar_norm"),
        ("Persistent GDN", "core_ar_norm", "core_gdn"),
        ("BA overlap", "core_gdn", "core_ba"),
    ],
    "producer": [
        ("SwiGLU/MXFP8 fusion", "core_ba", "core_swiglu"),
        ("GDN output norm/MXFP8 fusion", "core_swiglu", "core_gdn_quant"),
        ("AR/RMSNorm/MXFP8 fusion", "core_gdn_quant", "core_ar_quant"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp8-baseline", type=Path, required=True,
                        help="Benchmark results from the user's independently started FP8 service")
    parser.add_argument("--stock-aligned-group",
                        help="Optional group directory containing core_stock_aligned")
    args = parser.parse_args()
    groups, comparisons = dict(GROUPS), dict(COMPARISONS)
    if args.stock_aligned_group:
        groups[args.stock_aligned_group] = ["fp8", "core_stock_aligned"]
        comparisons[args.stock_aligned_group] = [
            ("FP8-path-aligned MXFP6 vs user FP8", "fp8", "core_stock_aligned")]
    result = collect(args.results, groups, comparisons, factorial=False,
                     baseline_root=args.fp8_baseline)
    baseline = result["groups"]["runtime"]["arms"]["fp8"]["launch"]
    if baseline.get("origin") != "user-started independent service":
        raise ValueError("Final FP8 baseline must be the user's independently started service")
    result["scope"] = (
        "Current Dense base Mach cumulative ablations; 3k/1k frozen ShareGPT HTTP, "
        "single sweep per arm. Two retained Mach GPU pairs run concurrently. "
        "One user-started independent FP8 service supplies the common baseline; "
        "assistant-launched FP8 diagnostic runs are excluded. Mach-to-Mach increments "
        "compare adjacent arms on the same pair. Repeated bridge arms must not be mixed across pairs. "
        "No throughput confidence intervals. All MXFP6 arms use the same pinned "
        "extension; this does not isolate the historical scale-initialization update. "
        "Both FP8 and MXFP6 eager experiments were cancelled at the user's request "
        "and the format-graph group is excluded. CUDA Graphs remain enabled "
        "throughout; no additional Mach gain is attributed to enabling graphs. "
        "MXFP6 vs the user FP8 baseline compares graph-enabled serving profiles, not "
        "quantization alone. The cancelled third group ran concurrently during "
        "the initial part of the retained measurements."
    )
    if args.stock_aligned_group:
        result["scope"] += (
            " Additional path-aligned MXFP6 restores stock compilation, graph policy, "
            "attention and serving defaults, and pins CUSTOM/PYNCCL AllReduce to "
            "match the user startup log. NCCL differs: user 2.30.7, Mach 2.29.7. It is measured "
            "later on GPU 2/3. Original core_graph is an optimization-disabled "
            "diagnostic floor, not a fair format baseline. MXFP8 quantization "
            "remains opaque to stock FP8 fusion patterns; alignment is not complete "
            "kernel-fusion equivalence."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report = markdown(result).replace("# Dense independent serving ablations",
                                      "# Dense base Mach cumulative serving ablations", 1)
    args.output.with_suffix(".md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
