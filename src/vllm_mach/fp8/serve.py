# SPDX-License-Identifier: Apache-2.0
"""Launch the explicit Qwen3.5-4B block-FP8 serving contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .profile import FEATURES, NAME


def configure_environment(directory, *, features=FEATURES, compile_mode="production"):
    features = tuple(features)
    if not features or len(set(features)) != len(features) or set(features) - set(FEATURES):
        raise ValueError(f"Select unique features from {FEATURES}")
    if compile_mode not in ("production", "quality4", "quality32", "quality64"):
        raise ValueError("Unknown block-FP8 compile mode")
    directory = Path(directory).resolve()
    cache = directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    startup = Path(tempfile.mkdtemp(prefix="startup-", dir=cache))
    settings = {
        "VLLM_MACH_PROFILE": NAME, "VLLM_PLUGINS": "mach",
        "VLLM_MACH_FP8_RUN_DIR": str(directory), "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_MACH_FP8_COMPILE_MODE": compile_mode,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "1", "VLLM_MACH_NATIVE_MXFP8": "0",
        "VLLM_QWEN3_5_FUSED_AR_NORM": "0", "VLLM_VOCAB_PARALLEL_GREEDY": "0",
        "VLLM_HYBRID_NVFP4_LM_HEAD": "0", "VLLM_HYBRID_MXFP8_LM_HEAD": "0",
        "VLLM_CACHE_ROOT": str(startup / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(startup / "inductor"),
        "TRITON_CACHE_DIR": str(startup / "triton"),
        **{f"VLLM_MACH_FP8_{name.upper()}": str(int(name in features)) for name in FEATURES},
    }
    os.environ.update(settings)
    return settings


def build_argv(model, *, tokenizer=None, host="127.0.0.1", port=8000,
               served_model_name="q35-fp8-study"):
    return ["vllm", "serve", str(model), "--tokenizer", str(tokenizer or model),
            "--served-model-name", served_model_name, "--host", host, "--port", str(port),
            "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--language-model-only",
            "--max-model-len", "8192", "--max-num-seqs", "128",
            "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.85",
            "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float32",
            "--attention-backend", "FLASH_ATTN", "--quantization", "compressed-tensors",
            "--linear-backend", "cutlass", "--kv-cache-memory", str(19 * 2**30),
            "--seed", "20260907"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="q35-fp8-study")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    command = build_argv(args.model.resolve(), tokenizer=args.tokenizer,
                         host=args.host, port=args.port, served_model_name=args.served_model_name)
    if args.print_command:
        print(json.dumps({"profile": NAME, "features": args.features, "argv": command}, indent=2))
        return
    if not (args.model / "config.json").is_file():
        parser.error("Model must be a local block-FP8 checkpoint directory")
    environment = configure_environment(args.run_dir, features=args.features)
    from .profile import check_environment, check_runtime_sources
    versions, sources = check_environment(), check_runtime_sources()
    receipt = {"profile": NAME, "argv": command, "environment": environment,
               "packages": versions, "sources": sources}
    path = args.run_dir.resolve() / f"launch-{os.getpid()}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
