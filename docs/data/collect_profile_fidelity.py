"""Collect matched Dense/MoE, M4/M32 fidelity runs for current serving profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from collect_native_comparison import bootstrap, read

HERE = Path(__file__).resolve().parent
ARMS = ("fp8", "nvfp4", "default", "full")


def physical_audit(source, samples, rows):
    """Validate both worker traces, including the repeated first cohort."""
    cohorts = len(samples) // rows
    steps = {0: 0, 1: 0}
    for index in range(cohorts + 1):
        start = (index % cohorts) * rows
        batch = samples[start : start + rows]
        expected_ids = {s["id"] for s in batch}
        max_gold = max(s["target_tokens"] for s in batch)
        observed = read(source / f"batch{index:02}.json")["runtime_calls"]
        assert {rank["rank"] for rank in observed} == {0, 1}
        for rank in observed:
            scored = [c for c in rank["calls"] if 1 <= c["offset"] <= max_gold]
            assert len(scored) == max_gold
            assert {c["offset"] for c in scored} == set(range(1, max_gold + 1))
            assert all(
                c["num_tokens"] == c["num_reqs"] == rows
                and set(c["ids"]) == expected_ids
                for c in scored
            )
            steps[rank["rank"]] += len(scored)
    return {
        "cohorts_including_repeat": cohorts + 1,
        "physical_rows": rows,
        "scored_decode_steps_per_rank": steps,
    }


def collect_run(source, samples, reference, rows, arm):
    ids = [s["id"] for s in samples]
    complete = read(source / "COMPLETE.json")
    assert complete["queries"] == len(samples) == 256
    assert complete["target_tokens"] == 10479
    records = read(source / "records.json")
    assert [r["id"] for r in records] == ids
    contract = read(source / "contract.json")
    assert contract["arm"] == arm
    assert contract["physical_rows"] == rows
    assert contract["profile_version"] == "current"
    errors = []
    for ref, record, sample in zip(reference, records, samples, strict=True):
        assert (
            len(ref["gold_logprobs"])
            == len(record["gold_logprobs"])
            == sample["target_tokens"]
        )
        delta = np.asarray(record["gold_logprobs"]) - ref["gold_logprobs"]
        assert np.isfinite(delta).all()
        errors.append(float(np.abs(delta).mean()))
    result = {
        "mae": float(np.mean(errors)),
        "ci95": bootstrap(errors),
        "query_mae": errors,
        "gold_logprobs": [r["gold_logprobs"] for r in records],
        "repeat": read(source / "repeat.json"),
        "repeat_gold_logprobs": [
            r["gold_logprobs"]
            for r in read(source / f"batch{len(samples) // rows:02}.json")["records"]
        ],
        "contract": contract,
        "physical_batch_audit": physical_audit(source, samples, rows),
    }
    if arm in ("default", "full"):
        result["dispatch"] = read(source / "gdn.json")
    return result


def collect(root, manifest):
    samples = read(manifest)["samples"]
    ids = [s["id"] for s in samples]
    models = {}
    for family in ("dense", "moe"):
        shapes = {}
        for rows in (32, 4):
            base = root / family / f"m{rows}"
            reference = read(base / "bf16/records.json")
            assert [r["id"] for r in reference] == ids
            runs = {}
            for arm in ("bf16", *ARMS):
                source = base / arm
                runs[arm] = collect_run(source, samples, reference, rows, arm)
            paired = (
                np.asarray(runs["full"]["query_mae"]) - runs["default"]["query_mae"]
            )
            shapes[f"m{rows}"] = {
                "physical_rows": rows,
                "runs": runs,
                "full_minus_default": {
                    "mean": float(paired.mean()),
                    "ci95": bootstrap(paired),
                },
            }
        models[family] = shapes
    return {
        "schema": "current-profile-fidelity/v1",
        "date": "2026-09-17",
        "metric": "Query-mean gold-token logprob MAE vs model- and shape-matched BF16",
        "query_count": 256,
        "target_tokens": 10479,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "queries": [{"id": s["id"], "domain": s["domain"]} for s in samples],
        "bootstrap": {"seed": 20260910, "replicates": 20000, "unit": "query"},
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "fidelity-samples.json")
    parser.add_argument(
        "--output", type=Path, default=HERE / "profile-fidelity-20260917.json"
    )
    parser.add_argument(
        "--moe-nvfp4-repeat",
        type=Path,
        help="Additional complete MoE M32 NVFP4 run directory",
    )
    args = parser.parse_args()
    data = collect(args.results, args.manifest)
    if args.moe_nvfp4_repeat:
        samples = read(args.manifest)["samples"]
        reference = read(args.results / "moe/m32/bf16/records.json")
        data["moe_nvfp4_m32_independent_repeat"] = collect_run(
            args.moe_nvfp4_repeat, samples, reference, 32, "nvfp4"
        )
    args.output.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    for family, shapes in data["models"].items():
        for shape, result in shapes.items():
            print(
                family,
                shape,
                {a: (r["mae"], r["ci95"]) for a, r in result["runs"].items()},
            )


if __name__ == "__main__":
    main()
