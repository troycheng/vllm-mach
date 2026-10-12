"""Measure synchronized TP1 decode batches and save their CUDA traces.

The default profiler samples 40 iterations after an 80-iteration delay.
The delay keeps chunked prefill out of BS32 decode samples; both are adjustable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_native_mxfp6 import request_one

MODEL = "/data1/models/Qwen3.5-4B-MXFP8"
ALIAS = "Qwen3.8-27B-MXFP6"
PROMPTS = json.loads((ROOT / "docs/data/serving-prompts.json").read_text())["prompts"]
ARMS = {
    "default": {},
    "b12x": {"VLLM_MACH_MXFP8_BACKEND": "flashinfer"},
    "gdn": {"VLLM_MACH_GDN_TP1": "1"},
    "gdn_flash_attn": {"VLLM_MACH_GDN_TP1": "1"},
    "gdn_nvfp4_head": {"VLLM_MACH_GDN_TP1": "1"},
    "nvfp4_head": {},
    "nvfp4_head_pdl": {"VLLM_MACH_MXFP8_PDL": "1"},
    "pdl": {"VLLM_MACH_MXFP8_PDL": "1"},
    "norm_quant": {"VLLM_MACH_MXFP8_NORM_QUANT": "1"},
    "norm_quant_pdl": {"VLLM_MACH_MXFP8_NORM_QUANT": "1",
                       "VLLM_MACH_MXFP8_PDL": "1"},
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


async def batch_request(session, base_url, batch, output_tokens):
    semaphore = asyncio.Semaphore(batch)
    begin = time.perf_counter()
    results = await asyncio.gather(
        *[
            request_one(
                session,
                semaphore,
                base_url + "/v1/completions",
                ALIAS,
                (
                    PROMPTS[i]["token_ids"]
                    * math.ceil(3000 / len(PROMPTS[i]["token_ids"]))
                )[:3000],
                output_tokens,
                42000 + i,
                i,
            )
            for i in range(batch)
        ]
    )
    wall_s = time.perf_counter() - begin
    if not all(row["success"] for row in results):
        raise RuntimeError(str([(r["http_status"], r["error"]) for r in results]))
    if not all(row["usage"]["prompt_tokens"] == 3000 and row["usage"]["completion_tokens"] == output_tokens
               for row in results):
        raise RuntimeError("Server token usage violates the fixed-token contract")
    # Text SSE chunks can group several tokens. TPOT is computed from token usage;
    # engine iteration timing and CUDA traces provide the fixed-BS measurements.
    result = {
        "wall_s": wall_s,
        "mean_ttft_ms": statistics.fmean(r["ttft_s"] * 1000 for r in results),
        "mean_tpot_ms": (
            statistics.fmean(r["tpot_s"] * 1000 for r in results)
            if output_tokens > 1
            else None
        ),
        "output_throughput_tokens_s": batch * output_tokens / wall_s,
        "batch": batch,
        "output_tokens": output_tokens,
        "requests": results,
    }
    for row in results:
        row["response_sha256"] = hashlib.sha256(
            row.pop("response_text").encode()
        ).hexdigest()
    return result


async def collect(args, arm):
    dest = args.output / arm
    base_url = f"http://127.0.0.1:{args.port}"
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in args.batches:
            print(f"START {arm} BS{batch}", flush=True)
            if args.phase == "prefill":
                await batch_request(session, base_url, batch, 1)
                before = set(dest.glob("*.pt.trace.json.gz"))
                async with session.post(base_url + "/start_profile") as response:
                    response.raise_for_status()
                result = await batch_request(session, base_url, batch, 1)
                async with session.post(base_url + "/stop_profile") as response:
                    response.raise_for_status()
                new = set(dest.glob("*.pt.trace.json.gz")) - before
                if len(new) != 1:
                    raise RuntimeError(f"Expected one prefill trace: {new}")
                new.pop().rename(dest / f"bs{batch}.pt.trace.json.gz")
                write_json(dest / f"bs{batch}-profile-request.json", result)
                print(f"COMPLETE {arm} BS{batch} prefill", flush=True)
                continue
            await batch_request(session, base_url, batch, args.warmup_tokens)
            samples = []
            for repeat in range(args.repeats):
                begin = time.time()
                result = await batch_request(session, base_url, batch, args.output_tokens)
                result["begin_epoch"] = begin
                result["end_epoch"] = time.time()
                result["repeat"] = repeat
                samples.append(result)
                write_json(dest / f"bs{batch}-serving.json", samples)
            if args.serving_only:
                print(
                    f"COMPLETE {arm} BS{batch}: "
                    f"{statistics.fmean(r['output_throughput_tokens_s'] for r in samples):.1f} tokens/s; "
                    f"mean TPOT {statistics.fmean(r['mean_tpot_ms'] for r in samples):.3f} ms",
                    flush=True,
                )
                continue
            before = set(dest.glob("*.pt.trace.json.gz"))
            async with session.post(base_url + "/start_profile") as response:
                response.raise_for_status()
            profile_request = await batch_request(session, base_url, batch, args.output_tokens)
            async with session.post(base_url + "/stop_profile") as response:
                response.raise_for_status()
            after = set(dest.glob("*.pt.trace.json.gz"))
            new = after - before
            if len(new) != 1:
                raise RuntimeError(f"Expected one trace for {arm} BS{batch}: {new}")
            trace = new.pop()
            renamed = dest / f"bs{batch}.pt.trace.json.gz"
            if renamed.exists():
                raise FileExistsError(renamed)
            trace.rename(renamed)
            write_json(dest / f"bs{batch}-profile-request.json", profile_request)
            print(
                f"COMPLETE {arm} BS{batch}: mean TPOT "
                f"{statistics.fmean(r['mean_tpot_ms'] for r in samples):.3f} ms; {renamed}",
                flush=True,
            )


def server_command(args, arm):
    config = {
        "profiler": "torch",
        "torch_profiler_dir": str(args.output / arm),
        "torch_profiler_with_stack": False,
        "delay_iterations": args.profile_delay if args.phase == "decode" else 0,
        "max_iterations": args.profile_steps if args.phase == "decode" else 0,
        "ignore_frontend": True,
    }
    command = [
        sys.executable,
        "-m",
        "vllm_mach.mxfp6.serve",
        "--model",
        MODEL,
        "--tensor-parallel-size",
        "1",
        "--served-model-name",
        ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--enable-logging-iteration-details",
        "--compilation-config",
        "{}",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--profiler-config",
        json.dumps(config),
    ]
    if arm == "gdn_flash_attn":
        command += [
            "--attention-backend",
            "FLASH_ATTN",
            "--attention-config",
            '{"flash_attn_version":2}',
        ]
    if arm in ("gdn_nvfp4_head", "nvfp4_head", "nvfp4_head_pdl"):
        command += ["--nvfp4-lm-head"]
    return command


def run(args):
    for arm in args.arms:
        dest = args.output / arm
        dest.mkdir(parents=True, exist_ok=True)
        if args.existing:
            asyncio.run(collect(args, arm))
            continue
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.device,
                "PYTHONPATH": str(ROOT / "src"),
                "VLLM_MACH_MXFP8_BACKEND": "native",
                "VLLM_MACH_GDN_TP1": "0",
                "VLLM_MACH_MXFP8_FUSED_MLP": "1",
                "VLLM_MACH_FUSED_GEMMA_NORM": "1",
                "VLLM_MACH_MXFP8_PDL": "0",
                "VLLM_MACH_MXFP8_NORM_QUANT": "0",
            }
        )
        env.update(ARMS[arm])
        command = server_command(args, arm)
        write_json(
            dest / "launch.json",
            {
                "command": command,
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith("VLLM_MACH_")
                    or k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MXFP8_LIBRARY_PATH",
                             "MXFP6_LIBRARY_PATH")
                },
                "contract": {"input_tokens": 3000, "output_tokens": args.output_tokens,
                             "warmup_tokens": args.warmup_tokens, "repeats": args.repeats,
                             "profile_delay": args.profile_delay,
                             "profile_steps": args.profile_steps,
                             "kv_cache_dtype": args.kv_cache_dtype,
                             "serving_only": args.serving_only},
            },
        )
        with (args.output / f"{arm}-server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                for _ in range(900):
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"{arm} exited: inspect {args.output / (arm + '-server.log')}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{args.port}/health", timeout=1
                        ):
                            break
                    except (OSError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError("server startup")
                asyncio.run(collect(args, arm))
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8253)
    parser.add_argument("--device", default="2")
    parser.add_argument(
        "--arms", nargs="+", choices=ARMS, default=["default", "b12x", "gdn"]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=1000)
    parser.add_argument("--warmup-tokens", type=int, default=1000)
    parser.add_argument("--profile-delay", type=int, default=80)
    parser.add_argument("--profile-steps", type=int, default=40)
    parser.add_argument("--kv-cache-dtype", default="auto",
                        choices=["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
                                 "fp8_per_token_head"])
    parser.add_argument("--serving-only", action="store_true",
                        help="Measure HTTP throughput without collecting CUDA traces")
    parser.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
