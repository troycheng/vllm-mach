#!/usr/bin/env python3
"""Alternate baseline/new-head server lifecycles on identical streaming workloads."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--devices", default="0,1")
    p.add_argument(
        "--profile", choices=("native", "mxfp8-2b", "mxfp8-4b"), default="native"
    )
    p.add_argument("--concurrency", type=int, nargs="+", default=[4, 32])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--input-tokens", type=int, default=1024)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("top20", 20, 0.95, 0.0),
        ("top20_presence", 20, 0.95, 1.5),
        ("top64", 64, 0.95, 0.0),
        ("full", -1, 1.0, 0.0),
    ]
    all_results = []
    for iteration in range(a.rounds):
        arms = (
            ["baseline", "optimized"]
            if iteration % 2 == 0
            else ["optimized", "baseline"]
        )
        for arm in arms:
            directory = a.output / f"{iteration}-{arm}"
            directory.mkdir()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=a.devices,
                OMP_NUM_THREADS="1",
                PYTHONPATH=str(root / "src"),
                VLLM_CACHE_ROOT=str(a.output / "cache"),
                VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=str(a.output / "autotune"),
            )
            command = [
                sys.executable,
                "-m",
                "vllm_mach.mxfp6.serve",
                "--model",
                str(a.model),
                "--nvfp4-lm-head",
                "--no-lossless-prefill",
                "--no-owner-prefill",
                "--no-fused-ar-quant",
                "--host",
                "127.0.0.1",
                "--port",
                str(a.port),
                "--served-model-name",
                "head-test",
                "--max-model-len",
                "4096",
                "--kv-cache-memory-bytes",
                "4294967296",
            ]
            if a.profile != "native":
                module = (
                    "vllm_mach.mxfp8.two_b.serve"
                    if a.profile == "mxfp8-2b"
                    else "vllm_mach.mxfp8.serve"
                )
                command = [
                    sys.executable,
                    "-m",
                    module,
                    str(a.model),
                    "--run-dir",
                    str(directory / "runtime"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(a.port),
                    "--served-model-name",
                    "head-test",
                ]
                if a.profile == "mxfp8-2b":
                    command += ["--attention-backend", "TRITON_ATTN", "--hybrid-head"]
            command += ["--hybrid-head" if arm == "optimized" else "--no-hybrid-head"]
            log = (directory / "server.log").open("w")
            proc = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 1800
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited: {directory}")
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{a.port}/health", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Server startup: {directory}")
                    time.sleep(2)
                print("READY", iteration, arm, flush=True)
                for name, k, topp, presence in cases:
                    for concurrency in a.concurrency:
                        dest = directory / f"{name}-c{concurrency}.json"
                        bench = [
                            sys.executable,
                            str(root / "tools/benchmark_native_mxfp6.py"),
                            "--base-url",
                            f"http://127.0.0.1:{a.port}",
                            "--model",
                            "head-test",
                            "--json-out",
                            str(dest),
                            "--num-prompts",
                            str(concurrency * 2),
                            "--input-tokens",
                            str(a.input_tokens),
                            "--output-tokens",
                            str(a.output_tokens),
                            "--contract-seed",
                            "20261015",
                            "--request-seed-base",
                            "2026101500",
                            "--max-concurrency",
                            str(concurrency),
                            "--warmup-requests",
                            str(concurrency),
                            "--warmup-output-tokens",
                            "32",
                            "--temperature",
                            ".7",
                            "--unseeded",
                            "--top-k",
                            str(k),
                            "--top-p",
                            str(topp),
                            "--presence-penalty",
                            str(presence),
                        ]
                        with (directory / f"{name}-c{concurrency}.log").open(
                            "w"
                        ) as stream:
                            subprocess.run(
                                bench,
                                check=True,
                                cwd=root,
                                env=env,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                            )
                        result = json.loads(dest.read_text())
                        assert result["aggregate"]["completed"] == concurrency * 2
                        all_results.append(
                            {
                                "iteration": iteration,
                                "profile": a.profile,
                                "arm": arm,
                                "case": name,
                                "concurrency": concurrency,
                                "contract": result["contract"],
                                "aggregate": result["aggregate"],
                            }
                        )
                        (a.output / "summary.json").write_text(
                            json.dumps(all_results, indent=2) + "\n"
                        )
                        print(
                            "COMPLETE",
                            iteration,
                            arm,
                            name,
                            concurrency,
                            result["aggregate"]["output_throughput_tokens_per_s"],
                            flush=True,
                        )
                if arm == "optimized":
                    assert (
                        "Using Mach stochastic LM-head path:"
                        in (directory / "server.log").read_text()
                    )
            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                log.close()
            time.sleep(3)


if __name__ == "__main__":
    main()
