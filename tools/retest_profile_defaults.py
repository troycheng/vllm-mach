"""Remeasure Dense default and MoE full under their archived README contracts."""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vllm_mach.mxfp6.serve import build_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="6,7")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["dense_default", "moe_full"],
        default=["dense_default", "moe_full"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        moe = arm == "moe_full"
        model = Path("/data1/models") / (
            "Qwen3.5-35B-A3B-MXFP6" if moe else "Qwen3.8-27B-MXFP6"
        )
        profile = argparse.Namespace(
            model=model,
            fp16_ssm=moe,
            lossless_prefill=None,
            owner_prefill=None,
            nvfp4_lm_head=moe,
            verify_prefill=False,
        )
        extra = [
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--served-model-name",
            "comparison",
            "--kv-cache-memory-bytes",
            "8589934592" if moe else "8218214400",
        ]
        if moe:
            archived = json.loads(
                Path("docs/data/qwen35-default-full-20260917.json").read_text()
            )
            extra += [
                "--max-num-seqs",
                "64",
                "--cudagraph-capture-sizes",
                *map(str, archived["mach_configuration"]["cudagraph_capture_sizes"]),
            ]
        command, env = build_command(profile, extra)
        env.update(
            CUDA_VISIBLE_DEVICES=args.devices, OMP_NUM_THREADS="4" if moe else "1"
        )
        dest = args.output / arm
        dest.mkdir(exist_ok=False)
        (dest / "launch.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        k: v
                        for k, v in env.items()
                        if k.startswith(("VLLM_", "MXFP6_"))
                        or k
                        in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")
                    },
                },
                indent=2,
            )
        )
        with (dest / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                for _ in range(900):
                    if server.poll() is not None:
                        raise RuntimeError(f"{arm} exited; see {dest / 'server.log'}")
                    try:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{args.port}/health", timeout=1
                        ).close()
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise TimeoutError("Service startup")
                for repeat in range(1, 3 if moe else 2):
                    for c in (4, 16, 24, 32):
                        out = dest / f"c{c}-r{repeat}.json"
                        bench = [
                            sys.executable,
                            "tools/benchmark_native_mxfp6.py",
                            "--base-url",
                            f"http://127.0.0.1:{args.port}",
                            "--model",
                            "comparison",
                            "--num-prompts",
                            str(max(16, 2 * c) if moe else 5 * c),
                            "--input-tokens",
                            "3000",
                            "--output-tokens",
                            "1000",
                            "--max-concurrency",
                            str(c),
                            "--warmup-requests",
                            str(c if moe else 32),
                            "--warmup-output-tokens",
                            "128",
                            "--contract-seed",
                            "20260917" if moe else "20260915",
                            "--request-seed-base",
                            "2026091700" if moe else "2026091500",
                            "--json-out",
                            str(out),
                        ]
                        if not moe:
                            bench += [
                                "--prompt-manifest",
                                "docs/data/serving-prompts.json",
                                "--request-rate",
                                "100",
                                "--top-k",
                                "20",
                                "--top-p",
                                ".95",
                            ]
                        subprocess.run(bench, check=True)
                        result = json.loads(out.read_text())
                        aggregate = result["aggregate"]
                        n = result["contract"]["num_prompts"]
                        assert aggregate["completed"] == aggregate["requested"] == n
                        assert aggregate["completion_tokens"] == 1000 * n
                        assert aggregate["prompt_tokens"] == 3000 * n
                        print(
                            arm,
                            c,
                            repeat,
                            aggregate["output_throughput_tokens_per_s"],
                            flush=True,
                        )
            finally:
                server.terminate()
                server.wait(timeout=90)


if __name__ == "__main__":
    main()
