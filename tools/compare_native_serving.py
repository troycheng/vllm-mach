#!/usr/bin/env python3
"""Run matched 3k/1k curves; stock baselines must use an unpatched vLLM runtime."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vllm_mach.mxfp6.serve import build_command


def launch_configuration(a, arm):
    """Keep stock compiler defaults separate from the Mach profile."""
    full = arm in ("full", "full_ba", "full_gdn")
    suffix = {"fp8": "FP8-official", "nvfp4": "NVFP4"}.get(arm, "MXFP6")
    args = argparse.Namespace(
        model=a.models / f"Qwen3.8-27B-{suffix}",
        fp16_ssm=full,
        gdn_persistent=arm in ("persistent", "gdn", "full_gdn"),
        gdn_ba_overlap=arm in ("full_ba", "gdn", "full_gdn"),
        lossless_prefill=full,
        owner_prefill=full,
        nvfp4_lm_head=full,
        verify_prefill=False,
    )
    command, env = build_command(
        args,
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(a.port),
            "--served-model-name",
            "comparison",
            "--kv-cache-memory-bytes",
            "8218214400",
        ],
    )
    env.update(CUDA_VISIBLE_DEVICES=a.devices, OMP_NUM_THREADS="1")
    if arm in ("fp8", "nvfp4"):
        # Do not merely disable flags in a patched runtime: import official packages.
        env = {k: v for k, v in env.items() if not k.startswith(("VLLM_", "MXFP6_"))}
        env["PYTHONPATH"] = str(a.stock_runtime.resolve())
        env.update(
            VLLM_PLUGINS="",
            VLLM_ALLREDUCE_USE_FLASHINFER="0",
        )
        index = command.index("--quantization")
        del command[index : index + 2]
        # Keep the stock compiler/graph defaults and ordinary serving admission.
        for flag in [
            "--compilation-config",
            "--kv-cache-memory-bytes",
            "--generation-config",
            "--limit-mm-per-prompt",
        ]:
            index = command.index(flag)
            del command[index : index + 2]
        command[command.index("--max-num-seqs") + 1] = "64"
        command += ["--gpu-memory-utilization", "0.9"]
    return command, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms",
        nargs="+",
        choices=[
            "default",
            "gdn",
            "persistent",
            "full",
            "full_ba",
            "full_gdn",
            "fp8",
            "nvfp4",
        ],
        required=True,
    )
    p.add_argument("--models", type=Path, required=True)
    p.add_argument(
        "--stock-runtime",
        type=Path,
        required=True,
        help="Directory containing clean official vllm/ and flashinfer/ packages",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--devices", default="0,1")
    p.add_argument("--port", type=int, default=8267)
    p.add_argument(
        "--requests",
        type=int,
        default=0,
        help="0 uses five request waves per concurrency",
    )
    p.add_argument("--prompt-manifest", type=Path, required=True)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[4, 16, 24, 32])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for arm in a.arms:
        dest = a.output / arm
        dest.mkdir(exist_ok=False)
        command, env = launch_configuration(a, arm)
        if arm in ("fp8", "nvfp4"):
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import inspect,vllm; from vllm.model_executor.models import qwen3_5; "
                    "assert 'VLLM_QWEN3_5_FUSED_AR_NORM' not in inspect.getsource(qwen3_5); "
                    "print('STOCK_RUNTIME',vllm.__file__)",
                ],
                env=env,
                check=True,
            )
        (dest / "launch.json").write_text(
            json.dumps(
                dict(
                    command=command,
                    environment={
                        k: v
                        for k, v in env.items()
                        if k.startswith(
                            ("VLLM_", "MXFP6_", "CUDA_VISIBLE", "PYTHONPATH")
                        )
                    },
                ),
                indent=2,
            )
        )
        with (dest / "server.log").open("x") as log:
            server = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print("START", arm, server.pid, flush=True)
            try:
                for _ in range(900):
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"{arm} failed to start: {server.returncode}"
                        )
                    try:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{a.port}/health", timeout=1
                        ).close()
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise TimeoutError("Service startup")
                for c in a.concurrencies:
                    command = [
                        sys.executable,
                        str(root / "tools/benchmark_native_mxfp6.py"),
                        "--base-url",
                        f"http://127.0.0.1:{a.port}",
                        "--model",
                        "comparison",
                        "--num-prompts",
                        str(a.requests or 5 * c),
                        "--input-tokens",
                        "3000",
                        "--output-tokens",
                        "1000",
                        "--max-concurrency",
                        str(c),
                        "--warmup-requests",
                        "32",
                        "--warmup-output-tokens",
                        "128",
                        "--contract-seed",
                        "20260915",
                        "--request-seed-base",
                        "2026091500",
                        "--json-out",
                        str(dest / f"c{c}.json"),
                        "--prompt-manifest",
                        str(a.prompt_manifest.resolve()),
                        "--request-rate",
                        "100",
                        "--top-k",
                        "20",
                        "--top-p",
                        ".95",
                    ]
                    with (dest / f"c{c}.log").open("x") as blog:
                        subprocess.run(
                            command,
                            stdout=blog,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=1800,
                        )
                    result = json.loads((dest / f"c{c}.json").read_text())["aggregate"]
                    print(
                        "RESULT",
                        arm,
                        c,
                        result["output_throughput_tokens_per_s"],
                        flush=True,
                    )
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)


if __name__ == "__main__":
    main()
