"""Validate and collect the September 17 Dense-default / MoE-full retest."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    if path.name.startswith("qwen35-"):
        text = json.dumps(data, indent=2, allow_nan=False)
        text = re.sub(
            r"\[\s+([^\[\]{}]+?)\s+\]",
            lambda match: json.dumps(json.loads(match.group()), separators=(",", ":")),
            text,
        )
    else:
        text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    assert json.loads(text) == data
    path.write_text(text + "\n")


def checked(path, contract):
    run = read(path)
    assert run["contract"] == contract, path
    n = contract["num_prompts"]
    a = run["aggregate"]
    assert a["completed"] == a["requested"] == n
    assert a["prompt_tokens"] == n * 3000
    assert a["completion_tokens"] == n * 1000
    assert len(run["requests"]) == n
    for row in run["requests"]:
        assert row["success"] and row["http_status"] == 200
        assert row["usage"]["prompt_tokens"] == 3000
        assert row["usage"]["completion_tokens"] == 1000
    return run


def rows(run, columns):
    return [
        [
            row["usage"][k]
            if k in ("prompt_tokens", "completion_tokens")
            else row["request_index"]
            if k == "index"
            else row[k]
            for k in columns
        ]
        for row in run["requests"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    dense = read(HERE / "native-serving.json")
    points = []
    for c in (4, 16, 24, 32):
        run = checked(
            args.results / "dense_default" / f"c{c}-r1.json", dense["contracts"][str(c)]
        )
        points.append(
            dict(
                concurrency=c,
                **run["aggregate"],
                warmup=run["warmup"],
                benchmark_started_epoch=run["benchmark_started_epoch"],
                request_rows=rows(run, dense["request_columns"]),
            )
        )
    dense["runs"]["prefill_default"] = {
        "date": "2026-09-17",
        "points": points,
        "launch": read(args.results / "dense_default" / "launch.json"),
        "description": "Current Dense default: GDN + lossless/owner prefill, FP32 SSM, BF16 head",
    }
    moe = read(HERE / "qwen35-default-full-20260917.json")
    moe["mach_configuration"].pop("ssm_dtype", None)
    moe["arms"]["default"]["ssm_dtype"] = "float32"
    if "full_fp32" not in moe["arms"]:
        moe["arms"]["full_fp32"] = dict(
            moe["arms"]["full"],
            ssm_dtype="float32",
            description="Archived full before FP16 SSM",
        )
    moe["arms"]["full"].update(
        ssm_dtype="float16",
        fp16_ssm=True,
        launch=read(args.results / "moe_full" / "launch.json"),
    )
    for point in moe["comparisons"]:
        c = point["concurrency"]
        if "full_fp32" not in point:
            point["full_fp32"] = point["full"]
        contract = point["default"]["repetitions"][0]["contract"]
        repetitions = []
        for repeat in (1, 2):
            run = checked(args.results / "moe_full" / f"c{c}-r{repeat}.json", contract)
            repetitions.append(
                {
                    "repeat": repeat,
                    **{
                        k: run[k]
                        for k in (
                            "contract",
                            "warmup",
                            "aggregate",
                            "benchmark_started_epoch",
                        )
                    },
                    "request_rows": rows(run, moe["request_columns"]),
                }
            )
        point["full"] = {
            "repetitions": repetitions,
            "aggregate": {
                k: mean(r["aggregate"][k] for r in repetitions)
                for k in ("output_throughput_tokens_per_s", "mean_ttft_ms")
            },
        }
    moe["protocol"]["fp16_full_retest"] = (
        "September 17: full with FP16 SSM, two sequential c4/c16/c24/c32 sweeps on GPUs 6/7; "
        "all request contracts unchanged. Default and user-provided baselines retained; "
        "previous full archived as full_fp32."
    )
    write(HERE / "native-serving.json", dense)
    write(HERE / "qwen35-default-full-20260917.json", moe)
    print("Validated 380 Dense and 320 MoE scored requests; collected all 12 points.")


if __name__ == "__main__":
    main()
