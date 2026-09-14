#!/usr/bin/env python3
"""Launch the complete checkpoint-hybrid profile without assembling old recipes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mxfp6-checkpoint", type=Path, required=True)
    parser.add_argument("--rank64-bundle", type=Path)
    parser.add_argument("--owner-prefill", action="store_true")
    parser.add_argument("--served-model-name", default="Qwen3.8-27B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, choices=(32, 48), default=48)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Local-only acceptance endpoints; not for a public service")
    parser.add_argument("--verify-owner-forwards", type=int, default=0)
    return parser.parse_args(argv)


def command(args):
    graphs = [1, 2, 4, 8, 16, 24, 32] + ([48] if args.max_num_seqs == 48 else [])
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(args.model),
            "--served-model-name", args.served_model_name, "--host", args.host, "--port", str(args.port),
            "--quantization", "exl3", "--dtype", "bfloat16", "--tensor-parallel-size", "2",
            "--max-model-len", "8192", "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", "4096", "--enable-chunked-prefill", "--no-enable-prefix-caching",
            "--kv-cache-dtype", "auto", "--kv-cache-memory-bytes", "8218214400",
            "--mamba-ssm-cache-dtype", "float16", "--attention-backend", "TRITON_ATTN",
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--generation-config", "vllm",
            "--compilation-config", json.dumps({"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                                               "cudagraph_capture_sizes": graphs})]


def main():
    args = arguments()
    for path in (args.model, args.mxfp6_checkpoint):
        if not (path / "config.json").is_file() or not (path / "model.safetensors.index.json").is_file():
            raise SystemExit(f"Missing model config/index: {path}")
        index = json.loads((path / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            shard = (path / name).resolve()
            if not shard.is_relative_to(path.resolve()) or not shard.is_file():
                raise SystemExit(f"Missing or unsafe model shard: {name}")
    env = dict(os.environ)
    # Eliminate stale flags from earlier experiments; this entry point owns the recipe.
    for key in list(env):
        if key.startswith(("EXL3_", "VLLM_MACH_", "VLLM_EXL3_", "B12X_")):
            del env[key]
    profile = ROOT / "profiles/vllm-0.29.0/qwen38-checkpoint-fp16-ssm.env"
    raw = subprocess.check_output(["bash", "-c", 'source "$1"; env -0', "mach-profile", str(profile)], env=env)
    env = dict(item.decode().split("=", 1) for item in raw.split(b"\0") if item)
    env["VLLM_MACH_MXFP6_CHECKPOINT"] = str(args.mxfp6_checkpoint)
    if args.rank64_bundle:
        env["VLLM_MACH_RANK64_BUNDLE"] = str(args.rank64_bundle)
    if args.owner_prefill:
        env["VLLM_MACH_OWNER_PREFILL"] = "1"
        env["VLLM_MACH_OWNER_VERIFY"] = str(args.verify_owner_forwards)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    launch = command(args)
    if args.diagnostics:
        if args.host not in ("127.0.0.1", "::1"):
            raise SystemExit("--diagnostics requires --host 127.0.0.1 or ::1")
        env["PYTHONPATH"] = str(ROOT / "tools")
        launch += ["--worker-extension-cls", "service_probe.WorkerExtension",
                   "--middleware", "service_probe.Middleware"]
    print(shlex.join(launch), flush=True)
    if args.dry_run:
        print(json.dumps({k: v for k, v in env.items() if k.startswith(
            ("EXL3_", "VLLM_MACH_", "VLLM_EXL3_", "B12X_", "MXFP6_"))}, indent=2))
        return
    os.execvpe(launch[0], launch, env)


if __name__ == "__main__":
    main()
