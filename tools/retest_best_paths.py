#!/usr/bin/env python3
"""Recheck qualified best paths against pre-sampling Mach, then sample modes.

Private launch paths and per-request responses stay in the requested output
directory. Reports should publish sanitized aggregates and content identities.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROFILES = (
    "2b-mxfp8",
    "4b-mxfp8",
    "4b-block-fp8",
    "27b-mxfp6",
    "35b-mxfp6",
    "27b-nvfp4",
    "35b-nvfp4",
)
CASES = {
    "greedy": (0.0, -1, 1.0, 0.0),
    "top20": (0.7, 20, 0.95, 0.0),
    "top20_presence": (0.7, 20, 0.95, 1.5),
    "top64": (0.7, 64, 0.95, 0.0),
    "topp": (0.7, -1, 0.95, 0.0),
    "full": (0.7, -1, 1.0, 0.0),
}
REQUESTS = {4: 40, 16: 160, 24: 120, 32: 160, 48: 240, 64: 320}


def check_idle(devices, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + devices,
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if all(
            int(row.split(",")[1].strip()) <= 64 for row in output.strip().splitlines()
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Requested GPU is occupied; no service was started: {output}"
            )
        time.sleep(3)


def command_probes(directory, op, **kwargs):
    results = []
    for path in sorted(directory.glob("rank-*.json")):
        receipt = json.loads(path.read_text())
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(30)
            client.connect(receipt["socket"])
            client.sendall(json.dumps({"op": op, **kwargs}).encode() + b"\n")
            response = json.loads(client.makefile("rb").readline())
            if "error" in response:
                raise RuntimeError(response)
            results.append(response)
    if not results:
        raise RuntimeError("No worker route probes found")
    return results


def run_bench(root, directory, a, env, case, concurrency, *, audit=False):
    temperature, k, p, penalty = CASES[case]
    small = a.profile.startswith(("2b", "4b"))
    count = (
        concurrency
        if audit
        else (REQUESTS[concurrency] if small else max(16, 2 * concurrency))
        if case == "greedy"
        else max(16, 2 * concurrency)
    )
    path = directory / f"{'audit-' if audit else ''}{case}-c{concurrency}.json"
    cmd = [
        sys.executable,
        str(root / "tools/benchmark_native_mxfp6.py"),
        "--base-url",
        f"http://127.0.0.1:{a.port}",
        "--model",
        "best-path",
        "--json-out",
        str(path),
        "--num-prompts",
        str(count),
        "--input-tokens",
        "3000",
        "--output-tokens",
        "256" if audit else "1000",
        "--contract-seed",
        "20261003" if audit else "20260924",
        "--request-seed-base",
        "2026100300" if audit else "2026092400",
        "--max-concurrency",
        str(concurrency),
        "--warmup-requests",
        "0" if audit else str(concurrency),
        "--warmup-output-tokens",
        "128",
        "--temperature",
        str(temperature),
        "--top-k",
        str(k),
        "--top-p",
        str(p),
        "--presence-penalty",
        str(penalty),
    ]
    if temperature > 0:
        cmd += ["--unseeded"]
    with path.with_suffix(".log").open("w") as stream:
        subprocess.run(
            cmd, check=True, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT
        )
    data = json.loads(path.read_text())
    ag = data["aggregate"]
    assert ag["completed"] == ag["requested"] == count
    assert ag["prompt_tokens"] == 3000 * count
    assert ag["completion_tokens"] == (256 if audit else 1000) * count
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8140)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    p.add_argument("--points", type=int, nargs="+")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    small = a.profile.startswith(("2b", "4b"))
    points = a.points or ([4, 16, 24, 32, 48, 64] if small else [4, 16, 24, 32])
    a.output.mkdir(parents=True, exist_ok=a.resume)
    summary = a.output / "summary.json"
    all_results = (
        json.loads(summary.read_text()) if a.resume and summary.exists() else []
    )
    completed = {
        (r["iteration"], r["arm"], r["case"], r["concurrency"]) for r in all_results
    }
    for iteration in range(a.rounds):
        for arm in (
            ["baseline", "current"] if iteration % 2 == 0 else ["current", "baseline"]
        ):
            if all(
                (iteration, arm, case, c) in completed
                for case in a.cases
                for c in points
            ):
                continue
            check_idle(a.devices)
            directory = a.output / f"{iteration}-{arm}"
            if directory.exists():
                generation = 1
                while (a.output / f"{iteration}-{arm}-resume{generation}").exists():
                    generation += 1
                directory = a.output / f"{iteration}-{arm}-resume{generation}"
            directory.mkdir()
            src = a.baseline_root if arm == "baseline" else root
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("VLLM_", "MXFP6_", "MX8_", "MACH_BENCH_"))
            }
            env.update(
                CUDA_VISIBLE_DEVICES=a.devices,
                OMP_NUM_THREADS="1",
                PYTHONPATH=os.pathsep.join([str(src / "src"), str(root / "tools")]),
                VLLM_CACHE_ROOT=str(directory / "cache/vllm"),
                TRITON_CACHE_DIR=str(directory / "cache/triton"),
                TORCHINDUCTOR_CACHE_DIR=str(directory / "cache/inductor"),
                VLLM_USE_FLASHINFER_SAMPLER="0",
            )
            cmd = [
                sys.executable,
                str(root / "tools/serve_best_path.py"),
                "--profile",
                a.profile,
                "--model",
                str(a.model),
                "--run-dir",
                str(directory / "runtime"),
                "--port",
                str(a.port),
            ]
            if arm == "current" and a.profile != "4b-block-fp8":
                cmd += ["--sampling"]
            (directory / "command.json").write_text(json.dumps(cmd, indent=2))
            stream = (directory / "server.log").open("w")
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            probes = directory / "runtime/probe"
            try:
                deadline = time.monotonic() + 2400
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"{a.profile} {arm} startup failed: {directory}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{a.port}/health", timeout=2
                        ):
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Service startup timed out")
                    time.sleep(2)
                print("READY", a.profile, iteration, arm, flush=True)
                initial = command_probes(probes, "snapshot")
                (directory / "initial-routes.json").write_text(
                    json.dumps(initial, indent=2)
                )
                for case in a.cases:
                    # Exercise the production default for every current case.
                    # Admission itself must exclude unfiltered/top-p-only calls.
                    enabled = arm == "current" and a.profile != "4b-block-fp8"
                    command_probes(probes, "set_sampling", enabled=enabled)
                    for c in points:
                        if (iteration, arm, case, c) in completed:
                            continue
                        command_probes(probes, "audit_on")
                        run_bench(root, directory, a, env, case, c, audit=True)
                        route = command_probes(probes, "audit_off")
                        assert all(not r["audit_active"] for r in route)
                        (directory / f"routes-{case}-c{c}.json").write_text(
                            json.dumps(route, indent=2)
                        )
                        compact = a.profile != "4b-block-fp8" and (
                            case == "greedy"
                            or (enabled and case.startswith("top") and case != "topp")
                        )
                        if compact:
                            assert all(
                                any(
                                    k.startswith("decode_hybrid_") and v > 0
                                    for k, v in r["counts"].items()
                                )
                                for r in route
                            ), route
                        result = run_bench(root, directory, a, env, case, c)
                        after = command_probes(probes, "snapshot")
                        assert all(not r["audit_active"] for r in after)
                        if case in ("full", "topp"):
                            assert all(
                                before["sampling"] == post["sampling"]
                                for before, post in zip(route, after, strict=True)
                            ), "Unbounded sampling entered the optimized path"
                        all_results.append(
                            {
                                "profile": a.profile,
                                "iteration": iteration,
                                "arm": arm,
                                "lifecycle": directory.name,
                                "case": case,
                                "concurrency": c,
                                "sampling_enabled": enabled,
                                "contract": result["contract"],
                                "aggregate": result["aggregate"],
                                "routes": route,
                                "post_timing_routes": after,
                            }
                        )
                        temporary = a.output / "summary.tmp"
                        temporary.write_text(json.dumps(all_results, indent=2) + "\n")
                        temporary.replace(a.output / "summary.json")
                        print(
                            "COMPLETE",
                            a.profile,
                            iteration,
                            arm,
                            case,
                            c,
                            result["aggregate"]["output_throughput_tokens_per_s"],
                            flush=True,
                        )
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                stream.close()
            time.sleep(3)


if __name__ == "__main__":
    main()
