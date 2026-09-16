"""Collect actual per-query fidelity and matched serving results for README plots."""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["fp8", "default", "persistent", "gdn", "full", "full_ba", "full_gdn", "nvfp4"]


def read(path):
    return json.loads(path.read_text())


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260910)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def fidelity(root, reuse_stock=False):
    reference = read(root / "fidelity/bf16/records.json")
    ids = [r["id"] for r in reference]
    assert len(ids) == len(set(ids)) == 256
    assert sum(len(r["gold_logprobs"]) for r in reference) == 10479
    archived = read(HERE / "native-fidelity-20260915.json") if reuse_stock else None
    if archived:
        assert ids == [q["id"] for q in archived["queries"]]
        assert [r["gold_logprobs"] for r in reference] == [
            r["bf16_logprobs"] for r in archived["queries"]
        ]
    runs = {}
    for arm in ARMS:
        if archived and arm in ("fp8", "nvfp4"):
            runs[arm] = archived["runs"][arm]
            runs[arm]["source"] = "native-fidelity-20260915.json"
            continue
        base = root / "fidelity" / arm
        assert read(base / "COMPLETE.json")["queries"] == 256
        records = read(base / "records.json")
        assert [r["id"] for r in records] == ids
        errors = []
        for ref, row in zip(reference, records, strict=True):
            assert len(ref["gold_logprobs"]) == len(row["gold_logprobs"])
            delta = np.asarray(row["gold_logprobs"]) - ref["gold_logprobs"]
            assert np.isfinite(delta).all()
            errors.append(float(np.abs(delta).mean()))
        runs[arm] = dict(
            mae=float(np.mean(errors)),
            ci95=bootstrap(errors),
            query_mae=errors,
            gold_logprobs=[r["gold_logprobs"] for r in records],
            repeat=read(base / "repeat.json"),
            contract=read(base / "contract.json"),
        )
    paired = np.asarray(runs["full"]["query_mae"]) - runs["default"]["query_mae"]
    head = read(root / "head/head.json")
    assert {r["rank"] for r in head} == {0, 1} and all(r["rows"] > 0 for r in head)
    assert head[0]["rows"] == head[1]["rows"]
    assert head[0]["global_argmax_mismatches"] == head[1]["global_argmax_mismatches"]
    head_rows = head[0]["rows"]
    missing20 = sum(r["global_top20_candidate_misses_on_rank"] for r in head)
    return dict(
        schema="native-mxfp6-fidelity/v1",
        query_count=256,
        target_tokens=10479,
        physical_rows=32,
        metric="Query-mean gold-token logprob MAE vs BF16",
        bootstrap=dict(seed=20260910, replicates=20000, unit="query"),
        reference_contract=read(root / "fidelity/bf16/contract.json"),
        reference_repeat=read(root / "fidelity/bf16/repeat.json"),
        queries=[
            dict(id=r["id"], domain=r["domain"], bf16_logprobs=r["gold_logprobs"])
            for r in reference
        ],
        runs=runs,
        full_minus_default=dict(mean=float(paired.mean()), ci95=bootstrap(paired)),
        gdn_ablation={
            arm: dict(
                reference=ref,
                mean=float(
                    np.mean(np.asarray(runs[arm]["query_mae"]) - runs[ref]["query_mae"])
                ),
                ci95=bootstrap(
                    np.asarray(runs[arm]["query_mae"]) - runs[ref]["query_mae"]
                ),
                exact_gold_logprobs=runs[arm]["gold_logprobs"]
                == runs[ref]["gold_logprobs"],
                dispatch=read(root / "fidelity" / arm / "gdn.json"),
            )
            for arm, ref in (
                ("persistent", "default"),
                ("gdn", "default"),
                ("full_ba", "full"),
                ("full_gdn", "full_ba"),
            )
        },
        head_probe=dict(
            scope="Free greedy decode, 256 corpus prompts × 48 tokens; same-hidden-state full BF16 reference",
            eligible_rows=head_rows,
            global_top20_tokens=head_rows * 20,
            global_top20_retained=head_rows * 20 - missing20,
            global_top20_recall=1 - missing20 / (head_rows * 20),
            final_top1_agreement=1 - head[0]["global_argmax_mismatches"] / head_rows,
            by_rank=head,
        ),
    )


def performance(root, reuse_stock=False):
    runs = {}
    contracts = {}
    archived = read(HERE / "native-serving-20260915.json") if reuse_stock else None
    for arm in ARMS:
        if archived and arm in ("fp8", "nvfp4"):
            runs[arm] = archived["runs"][arm]
            runs[arm]["source"] = "native-serving-20260915.json"
            contracts.update({int(c): v for c, v in archived["contracts"].items()})
            continue
        points = []
        for c in [4, 16, 24, 32]:
            data = read(root / "serving" / arm / f"c{c}.json")
            contract, aggregate = data["contract"], data["aggregate"]
            assert (
                aggregate["completed"]
                == aggregate["requested"]
                == contract["num_prompts"]
            )
            assert aggregate["prompt_tokens"] == contract["num_prompts"] * 3000
            assert aggregate["completion_tokens"] == contract["num_prompts"] * 1000
            assert contract["max_concurrency"] == c
            if c in contracts:
                assert contract == contracts[c]
            contracts[c] = contract
            columns = [
                "request_index",
                "success",
                "http_status",
                "prompt_tokens",
                "completion_tokens",
                "latency_s",
                "ttft_s",
                "tpot_s",
            ]
            rows = [
                [
                    (
                        r["usage"][k]
                        if k in ("prompt_tokens", "completion_tokens")
                        else r[k]
                    )
                    for k in columns
                ]
                for r in data["requests"]
            ]
            points.append(
                dict(
                    concurrency=c,
                    **aggregate,
                    warmup=data["warmup"],
                    benchmark_started_epoch=data["benchmark_started_epoch"],
                    request_rows=rows,
                )
            )
        runs[arm] = dict(
            points=points, launch=read(root / "serving" / arm / "launch.json")
        )
    return dict(
        schema="native-serving-comparison/v1",
        runs=runs,
        contracts=contracts,
        request_columns=columns,
        scope="Single-run short 3k/1k sweeps; no confidence interval for throughput",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--fidelity-only", action="store_true")
    p.add_argument(
        "--reuse-stock-baselines",
        action="store_true",
        help="Reuse archived September 15 FP8/NVFP4 data",
    )
    a = p.parse_args()
    data = fidelity(a.results, a.reuse_stock_baselines)
    (HERE / "native-fidelity.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print({n: (r["mae"], r["ci95"]) for n, r in data["runs"].items()})
    if not a.fidelity_only:
        (HERE / "native-serving.json").write_text(
            json.dumps(
                performance(a.results, a.reuse_stock_baselines),
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
