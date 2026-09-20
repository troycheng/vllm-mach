#!/usr/bin/env python3
"""Benchmark a user-started Dense FP8 service; never launch or modify a server."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices-note", default="not supplied")
    parser.add_argument("--server-command-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "origin": "user-started independent service",
        "base_url": args.base_url,
        "served_model": args.model,
        "declared_gpu_devices": args.devices_note,
        "server_started_or_modified_by_benchmark": False,
        "declared_server_command": args.server_command_file.read_text()
        if args.server_command_file else None,
    }
    (args.output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for c in (4, 16, 24, 32):
        command = [
            sys.executable, str(root / "tools/benchmark_native_mxfp6.py"),
            "--base-url", args.base_url, "--model", args.model,
            "--num-prompts", str(5 * c), "--input-tokens", "3000",
            "--output-tokens", "1000", "--max-concurrency", str(c),
            "--warmup-requests", "32", "--warmup-output-tokens", "128",
            "--contract-seed", "20260915", "--request-seed-base", "2026091500",
            "--json-out", str(args.output / f"c{c}.json"),
            "--prompt-manifest", str(root / "docs/data/serving-prompts.json"),
            "--request-rate", "100", "--top-k", "20", "--top-p", ".95",
        ]
        with (args.output / f"c{c}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        raw = json.loads((args.output / f"c{c}.json").read_text())
        agg = raw["aggregate"]
        assert agg["completed"] == agg["requested"] == 5 * c
        assert agg["prompt_tokens"] == 3000 * 5 * c
        assert agg["completion_tokens"] == 1000 * 5 * c
        print("RESULT", c, agg["output_throughput_tokens_per_s"], flush=True)


if __name__ == "__main__":
    main()
