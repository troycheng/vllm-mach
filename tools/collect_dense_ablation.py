#!/usr/bin/env python3
"""Validate and summarize two matched Dense 2x2 serving experiments."""

import argparse
import json
from pathlib import Path

GROUPS = {
    "prefill": ["fp8", "dense_none", "dense_lossless", "dense_owner", "dense_default"],
    "state-head": ["fp8", "dense_default", "dense_ssm", "dense_head", "dense_full"],
}
COMPARISONS = {
    "prefill": [
        ("Lossless alone", "dense_none", "dense_lossless"),
        ("Owner alone", "dense_none", "dense_owner"),
        ("Lossless with owner", "dense_owner", "dense_default"),
        ("Owner with lossless", "dense_lossless", "dense_default"),
        ("Both prefill options", "dense_none", "dense_default"),
    ],
    "state-head": [
        ("FP16 SSM alone", "dense_default", "dense_ssm"),
        ("NVFP4 head alone", "dense_default", "dense_head"),
        ("FP16 SSM with NVFP4 head", "dense_head", "dense_full"),
        ("NVFP4 head with FP16 SSM", "dense_ssm", "dense_full"),
        ("Both full options", "dense_default", "dense_full"),
    ],
}
CONCURRENCIES = (4, 16, 24, 32)
TPS = "output_throughput_tokens_per_s"


def collect(root, groups=GROUPS, comparisons=COMPARISONS, *, factorial=True, baseline_root=None):
    result = {
        "scope": "Current Dense, 3k/1k frozen ShareGPT HTTP; single sweep per arm; "
                 "two GPU pairs run concurrently, comparisons only within each pair. "
                 "No throughput confidence intervals.",
        "groups": {},
    }
    for name in ("provenance", "environment"):
        path = root / f"{name}.json"
        result[name] = json.loads(path.read_text()) if path.exists() else {"recorded": False}
    result["request_columns"] = ["request_index", "success", "http_status",
                                 "prompt_tokens", "completion_tokens", "latency_s", "ttft_s", "tpot_s"]
    for group, arms in groups.items():
        records, contracts = {}, {}
        for arm in arms:
            dest = baseline_root if arm == "fp8" and baseline_root is not None else root / group / arm
            record = {"launch": json.loads((dest / "launch.json").read_text()), "points": {}}
            for c in CONCURRENCIES:
                raw = json.loads((dest / f"c{c}.json").read_text())
                agg = raw["aggregate"]
                assert agg["completed"] == agg["requested"] == 5 * c, (group, arm, c)
                assert agg["prompt_tokens"] == 3000 * 5 * c
                assert agg["completion_tokens"] == 1000 * 5 * c
                assert abs(agg[TPS] - agg["completion_tokens"] / agg["duration_s"]) < 1e-7
                assert len(raw["requests"]) == 5 * c
                for request in raw["requests"]:
                    assert request["success"] and request["http_status"] == 200
                    assert request["usage"]["prompt_tokens"] == 3000
                    assert request["usage"]["completion_tokens"] == 1000
                matched = {k: raw[k] for k in ("contract", "warmup")}
                if c in contracts:
                    assert matched == contracts[c], ("contract mismatch", group, arm, c)
                contracts[c] = matched
                agg["request_rows"] = [
                    [r["request_index"], r["success"], r["http_status"],
                     r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"],
                     r["latency_s"], r["ttft_s"], r["tpot_s"]]
                    for r in raw["requests"]
                ]
                record["points"][str(c)] = agg
            records[arm] = record
        def rate(arm, c):
            return records[arm]["points"][str(c)][TPS]
        for arm in arms:
            gains = {str(c): 100 * (rate(arm, c) / rate("fp8", c) - 1)
                     for c in CONCURRENCIES}
            records[arm]["gain_over_fp8_pct"] = gains
            records[arm]["mean_gain_over_fp8_pct"] = sum(gains.values()) / len(gains)
        conditional = []
        for label, base, candidate in comparisons[group]:
            gains = {str(c): 100 * (rate(candidate, c) / rate(base, c) - 1)
                     for c in CONCURRENCIES}
            conditional.append(dict(label=label, base=base, candidate=candidate,
                                    gain_pct=gains, mean_gain_pct=sum(gains.values()) / len(gains)))
        result["groups"][group] = dict(arms=records, contracts=contracts,
                                      comparisons=conditional)
        if factorial:
            _, control, only_a, only_b, both = arms
            result["groups"][group]["multiplicative_interaction_pct"] = {
                str(c): 100 * (rate(both, c) * rate(control, c)
                              / (rate(only_a, c) * rate(only_b, c)) - 1)
                for c in CONCURRENCIES
            }
    return result


def markdown(result):
    lines = ["# Dense independent serving ablations", "", result["scope"], ""]
    for group, data in result["groups"].items():
        lines += [f"## {group}", "", "Output tokens/s (gain over the recorded FP8 baseline):", "",
                  "| Arm | c4 | c16 | c24 | c32 | Mean gain |", "|---|---:|---:|---:|---:|---:|"]
        for name, arm in data["arms"].items():
            values = [f"{arm['points'][str(c)][TPS]:.2f} ({arm['gain_over_fp8_pct'][str(c)]:+.2f}%)"
                      for c in CONCURRENCIES]
            lines.append(f"| {name} | " + " | ".join(values) + f" | {arm['mean_gain_over_fp8_pct']:+.2f}% |")
        lines += ["", "Conditional gains (candidate/base − 1):", "",
                  "| Comparison | c4 | c16 | c24 | c32 | Mean |", "|---|---:|---:|---:|---:|---:|"]
        for comp in data["comparisons"]:
            values = [f"{comp['gain_pct'][str(c)]:+.2f}%" for c in CONCURRENCIES]
            lines.append(f"| {comp['label']} | " + " | ".join(values) + f" | {comp['mean_gain_pct']:+.2f}% |")
        if "multiplicative_interaction_pct" in data:
            lines += ["", "Multiplicative interaction = both × control / (only A × only B) − 1; "
                      "descriptive only, not a significance test.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp8-baseline", type=Path,
                        help="Optional results from a user-started independent FP8 service")
    args = parser.parse_args()
    result = collect(args.results, baseline_root=args.fp8_baseline)
    if args.fp8_baseline is not None:
        launch = result["groups"]["prefill"]["arms"]["fp8"]["launch"]
        if launch.get("origin") != "user-started independent service":
            raise ValueError("External FP8 baseline must be the user's independently started service")
        result["scope"] = (
            "Earlier same-day Dense Mach 2x2 measurements re-expressed against "
            "the user's later independently started FP8 service. Identical 3k/1k "
            "request contracts verified. Mach increments remain within each "
            "original GPU pair; all FP8 denominators come from the external service. "
            "Single sweeps, unlocked clocks, no throughput confidence intervals."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
