#!/usr/bin/env python3
"""Publish sanitized paired serving results and the dated recheck tables."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from retest_best_paths import CASES

ROOT = Path(__file__).resolve().parents[1]
STEM = "best-path-recheck-20261015"
TITLES = {
    "2b-mxfp8": "Qwen3.5-2B MXFP8",
    "4b-mxfp8": "Qwen3.5-4B MXFP8",
    "4b-block-fp8": "Qwen3.5-4B block-FP8",
}


def route_receipt(route, *, after_timing=False):
    result = {k: v for k, v in route.items() if k not in ("pid", "socket")}
    if after_timing:
        # Audit counters stay unchanged while timing, so publish them once.
        result.pop("counts", None)
    return result


def confirmation(source):
    path = source / "4b-top20-c48-confirmation" / "summary.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    keys = {(r["iteration"], r["arm"]) for r in rows}
    assert len(keys) == len(rows) <= 4
    trials = []
    for r in rows:
        assert (
            r["profile"] == "4b-mxfp8"
            and r["case"] == "top20"
            and r["concurrency"] == 48
        )
        ag = r["aggregate"]
        assert ag["completed"] == ag["requested"] == 96
        assert ag["prompt_tokens"] == 288000 and ag["completion_tokens"] == 96000
        assert all(not x["audit_active"] for x in r["routes"] + r["post_timing_routes"])
        if r["arm"] == "current":
            for x in r["routes"]:
                assert any(
                    k.startswith("decode_hybrid_") and v > 0
                    for k, v in x["counts"].items()
                )
                assert not any(
                    k.startswith("decode_complete_head_") and v > 0
                    for k, v in x["counts"].items()
                )
        trials.append(
            {
                "iteration": r["iteration"],
                "arm": r["arm"],
                "aggregate": ag,
                "routes": [route_receipt(x) for x in r["routes"]],
            }
        )
    result = {
        "complete": keys == {(i, a) for i in range(2) for a in ("baseline", "current")},
        "trials": trials,
    }
    if result["complete"]:
        for arm in ("baseline", "current"):
            result[arm + "_tps"] = statistics.mean(
                r["aggregate"]["output_throughput_tokens_per_s"]
                for r in rows
                if r["arm"] == arm
            )
        result["gain_pct"] = 100 * (result["current_tps"] / result["baseline_tps"] - 1)
    return result


def collect(source):
    models = []
    for profile, title in TITLES.items():
        path = source / profile / "summary.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        points = (
            [4, 16, 24, 32, 48, 64]
            if profile.startswith(("2b", "4b"))
            else [4, 16, 24, 32]
        )
        expected = {
            (i, a, mode, c)
            for i in range(2)
            for a in ("baseline", "current")
            for mode in CASES
            for c in points
        }
        seen = set()
        trials = []
        for row in rows:
            key = (row["iteration"], row["arm"], row["case"], row["concurrency"])
            assert key in expected and key not in seen, (profile, key)
            seen.add(key)
            contract = row["contract"].copy()
            offsets = contract.pop("arrival_offsets_s")
            assert all(x == 0 for x in offsets)
            contract["arrival_offset_s"] = 0
            ag = row["aggregate"]
            assert ag["completed"] == ag["requested"] == contract["num_prompts"]
            assert ag["prompt_tokens"] == 3000 * ag["completed"]
            assert ag["completion_tokens"] == 1000 * ag["completed"]
            routes = [route_receipt(r) for r in row["routes"]]
            assert len(routes) == (1 if profile.startswith(("2b", "4b")) else 2)
            assert all(
                not r["audit_active"] for r in routes + row["post_timing_routes"]
            )
            compact = profile != "4b-block-fp8" and (
                row["case"] == "greedy"
                or (
                    row["sampling_enabled"]
                    and row["case"] in ("top20", "top20_presence", "top64")
                )
            )
            if compact:
                for r in routes:
                    assert (
                        sum(
                            v
                            for k, v in r["counts"].items()
                            if k.startswith("decode_hybrid_")
                        )
                        > 0
                    ), (profile, key, r)
                    assert (
                        sum(
                            v
                            for k, v in r["counts"].items()
                            if k.startswith("decode_complete_head_")
                        )
                        == 0
                    ), (profile, key, r)
            if compact and row["arm"] == "current" and row["case"] != "greedy":
                for before, after in zip(
                    routes, row["post_timing_routes"], strict=True
                ):
                    assert (
                        after["sampling"]["topk_calls"]
                        > before["sampling"]["topk_calls"]
                    ), (profile, key)
                    if profile in ("2b-mxfp8", "4b-mxfp8"):
                        assert not any(
                            (after.get("sampling_can_use_failures") or {}).values()
                        ), (profile, key)
            trials.append(
                {
                    k: row[k]
                    for k in (
                        "iteration",
                        "arm",
                        "case",
                        "concurrency",
                        "sampling_enabled",
                    )
                }
                | {
                    "lifecycle": row.get(
                        "lifecycle", f"{row['iteration']}-{row['arm']}"
                    ),
                    "contract": contract,
                    "aggregate": ag,
                    "routes": routes,
                    "post_timing_routes": [
                        route_receipt(r, after_timing=True)
                        for r in row["post_timing_routes"]
                    ],
                    "eligible_decode_hybrid_verified": compact,
                }
            )
        comparisons = []
        for mode in CASES:
            for c in points:
                arms = {
                    a: [
                        r
                        for r in trials
                        if r["case"] == mode and r["concurrency"] == c and r["arm"] == a
                    ]
                    for a in ("baseline", "current")
                }
                if any(len(v) != 2 for v in arms.values()):
                    continue
                assert all(
                    r["contract"] == arms["baseline"][0]["contract"]
                    for v in arms.values()
                    for r in v
                )
                result = {"case": mode, "concurrency": c}
                for a, runs in arms.items():
                    result[a] = {
                        k: statistics.mean(r["aggregate"][k] for r in runs)
                        for k in runs[0]["aggregate"]
                        if k.startswith(("mean_", "p99_"))
                        or k == "output_throughput_tokens_per_s"
                    }
                    result[a]["trial_tps"] = [
                        r["aggregate"]["output_throughput_tokens_per_s"] for r in runs
                    ]
                result["gain_pct"] = 100 * (
                    result["current"]["output_throughput_tokens_per_s"]
                    / result["baseline"]["output_throughput_tokens_per_s"]
                    - 1
                )
                comparisons.append(result)
        recipes = []
        for launch in sorted((source / profile).glob("*/runtime/launch.json")):
            receipt = json.loads(launch.read_text())
            lifecycle = launch.parent.parent.name
            scored = sum(
                r.get("lifecycle", f"{r['iteration']}-{r['arm']}") == lifecycle
                for r in rows
            )
            if not scored:
                continue
            argv = receipt["argv"][:]
            # The positional checkpoint and local probe/cache roots are private.
            argv = ["<checkpoint>" if arg.startswith("/") else arg for arg in argv]
            environment = {
                k: v
                for k, v in receipt["environment"].items()
                if not any(
                    part in k for part in ("DIR", "ROOT", "PATH", "VISIBLE_DEVICES")
                )
                and not str(v).startswith("/")
            }
            recipes.append(
                {
                    "lifecycle": lifecycle,
                    "scored_points": scored,
                    "argv": argv,
                    "environment": environment,
                    "packages": receipt["packages"],
                    "runtime_sources_sha256": receipt["sources"].get("sha256", {}),
                    "model_identity": {
                        k: v
                        for k, v in receipt["model"].items()
                        if k not in ("kv_scales", "restored_bf16_weights")
                    },
                }
            )
            if profile == "4b-block-fp8":
                worker = json.loads(
                    next(launch.parent.glob("worker-*.json")).read_text()
                )
                linear, activation, choices = (
                    worker[k] for k in ("linear", "activation", "compile_choices")
                )
                assert set(worker["features"]) == {"n64", "ordered", "silu", "fa2"}
                assert (
                    linear["counts"]["n64_calls"] > 0
                    and linear["counts"]["ordered_calls"] > 0
                )
                assert activation["counts"]["native"] > 0
                assert (
                    not choices["missing_bindings"] and not choices["source_mismatches"]
                )
                recipes[-1]["worker_startup_validation"] = {
                    "features": worker["features"],
                    "linear_counts": linear["counts"],
                    "activation_counts": activation["counts"],
                    "activation_sha256": activation["native_identity"]["sha256"],
                    "compile_policy_sha256": choices["policy_sha256"],
                    "required_bindings": choices["required_bindings"],
                    "binding_hits": choices["binding_hits"],
                    "note": "Eager/capture invocation counters; not CUDA Graph replay counts",
                }
        models.append(
            {
                "profile": profile,
                "status": "complete" if seen == expected else "pending",
                "title": title,
                "complete": seen == expected,
                "completed_points": len(seen),
                "expected_points": len(expected),
                "trials": trials,
                "comparisons": comparisons,
                "launch_recipes": recipes,
            }
        )
    return {
        "schema_version": 1,
        "date": "2026-10-15",
        "baseline_commit": "9d878ad0c35666faef3ddf1aca67b41016ccbc6e",
        "current_mach_sources_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/vllm_mach").rglob("*.py"))
        },
        "hardware": "RTX 5090",
        "complete": all(m["complete"] for m in models),
        "active_scope_complete": all(m["complete"] for m in models),
        "greedy_repeatability": json.loads(
            (source / "greedy-repeatability.json").read_text()
        )
        if (source / "greedy-repeatability.json").exists()
        else None,
        "top20_c48_confirmation": confirmation(source),
        "models": models,
    }


def charts(data, preview_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_performance_docs import CHART_THEME, themed_svg

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": STEM,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": CHART_THEME["text"][0],
            "axes.labelcolor": CHART_THEME["text"][0],
            "xtick.color": CHART_THEME["text"][0],
            "ytick.color": CHART_THEME["text"][0],
            "axes.edgecolor": CHART_THEME["axis"][0],
            "grid.color": CHART_THEME["grid"][0],
        }
    )
    dest = ROOT / "docs/assets"
    for model in data["models"]:
        if not model["complete"]:
            continue
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.4))
        rows = [r for r in model["comparisons"] if r["case"] == "greedy"]
        for arm, color, label in (
            ("baseline", "fp8", "Pre-sampling Mach"),
            ("current", "full", "Current best path"),
        ):
            left.plot(
                [r["concurrency"] for r in rows],
                [r[arm]["output_throughput_tokens_per_s"] for r in rows],
                marker="o",
                color=CHART_THEME[color][0],
                label=label,
            )
        left.set_title("Greedy output throughput")
        left.set_ylabel("Output tokens/s")
        for mode, color, marker in (
            ("top20", "full", "o"),
            ("top20_presence", "nvfp4", "s"),
            ("top64", "default", "^"),
        ):
            rows = [r for r in model["comparisons"] if r["case"] == mode]
            right.plot(
                [r["concurrency"] for r in rows],
                [r["gain_pct"] for r in rows],
                color=CHART_THEME[color][0],
                marker=marker,
                label=mode.replace("_", " + "),
            )
        right.axhline(0, color=CHART_THEME["axis"][0], linewidth=1)
        right.set_title(
            "Original-sampler throughput change"
            if model["profile"] == "4b-block-fp8"
            else "Bounded sampling throughput gain"
        )
        right.set_ylabel("Gain (%)")
        for ax in (left, right):
            xs = sorted({r["concurrency"] for r in model["comparisons"]})
            ax.set_xticks(xs, [f"c{x}" for x in xs])
            ax.set_xlabel("Concurrency")
            ax.grid(axis="y", alpha=0.55)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(model["title"] + " · 3,000 input / 1,000 output · two trials/arm")
        fig.tight_layout()
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.81, wspace=0.27)
        path = dest / f"{STEM}-{model['profile']}.svg"
        fig.savefig(
            path, metadata={"Date": None}, facecolor=CHART_THEME["background"][0]
        )
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                preview_dir / path.with_suffix(".png").name, dpi=130, facecolor="white"
            )
        plt.close(fig)
        path.write_text(
            "\n".join(line.rstrip() for line in themed_svg(path.read_text()).splitlines())
            + "\n"
        )


def report(data):
    lines = [
        "## Results",
        "",
        "Two lifecycle measurements per arm are required for every comparison below.",
        "Completed points include all six modes; unpaired points remain pending.",
        "",
        "| Profile | Completed points | Status |",
        "|---|---:|---|",
    ]
    for m in data["models"]:
        lines.append(
            f"| {m['title']} | {m['completed_points']}/{m['expected_points']} | {'Complete' if m['complete'] else 'Pending'} |"
        )
    for m in data["models"]:
        if not m["complete"]:
            continue
        lines += [
            "",
            f"### {m['title']}",
            "",
            f"![{m['title']} throughput and sampling gains](assets/{STEM}-{m['profile']}.svg)",
            "",
            "| Mode | Concurrency | Baseline tokens/s | Current tokens/s | Gain | Current TTFT ms | Current TPOT ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in m["comparisons"]:
            lines.append(
                f"| {r['case']} | c{r['concurrency']} | {r['baseline']['output_throughput_tokens_per_s']:.2f} | {r['current']['output_throughput_tokens_per_s']:.2f} | {r['gain_pct']:+.2f}% | {r['current']['mean_ttft_ms']:.2f} | {r['current']['mean_tpot_ms']:.3f} |"
            )
        worst = min(m["comparisons"], key=lambda r: r["gain_pct"])
        lines += [
            "",
            f"Worst measured change: **{worst['gain_pct']:+.2f}%**, {worst['case']} c{worst['concurrency']}.",
        ]
        if m["profile"] != "4b-block-fp8":
            lines += [
                "Eligible greedy and enabled bounded-sampling audits on every rank recorded",
                "hybrid decode and zero complete-head decode calls; prefill/mixed calls are excluded.",
            ]
    lines += [
        "",
        "### Independent 4B top-k20 c48 confirmation",
        "",
        "The original matrix retains its measured −1.08% mean at this point; one",
        "current trial was 4,189.18 tokens/s versus 4,658.22 in the other trial.",
        "A separate baseline/current/current/baseline recheck uses the same",
        "3,000/1,000 workload and flag selection. It does not replace the matrix.",
        "",
    ]
    extra = data["top20_c48_confirmation"]
    if extra["complete"]:
        lines += [
            "| Baseline tokens/s | Current tokens/s | Gain |",
            "|---:|---:|---:|",
            f"| {extra['baseline_tps']:.2f} | {extra['current_tps']:.2f} | {extra['gain_pct']:+.2f}% |",
            "",
        ]
    else:
        lines += [f"Pending: {len(extra['trials'])}/4 lifecycle measurements.", ""]
    lines += [
        f"[Machine-readable trials, latency and dispatch evidence](data/{STEM}.json).",
        "",
    ]
    path = ROOT / "docs" / f"{STEM}.md"
    text = path.read_text()
    text = re.sub(
        r"## Results\n.*?(?=## Reproduction)",
        "\n".join(lines) + "\n",
        text,
        flags=re.DOTALL,
    )
    if data["active_scope_complete"]:
        text = text.replace(
            "Measurements are being collected.",
            "The reported 2B/4B profiles have completed paired measurements.",
        )
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--preview-dir", type=Path)
    a = p.parse_args()
    data = collect(a.source)
    data["measurement_sha256"] = {
        m["profile"]: hashlib.sha256(
            (a.source / m["profile"] / "summary.json").read_bytes()
        ).hexdigest()
        for m in data["models"]
        if m["completed_points"]
    }
    output = ROOT / "docs/data" / f"{STEM}.json"
    output.write_text(json.dumps(data, indent=2) + "\n")
    if not a.no_plots:
        charts(data, a.preview_dir)
    report(data)
    print(
        json.dumps(
            {
                "complete": data["complete"],
                "progress": {
                    m["profile"]: m["completed_points"] for m in data["models"]
                },
            }
        )
    )


if __name__ == "__main__":
    main()
