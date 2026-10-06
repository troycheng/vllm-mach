# SPDX-License-Identifier: Apache-2.0
"""Launch the complete, versioned 4B MXFP8 profile from one checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .profile import NAME, quality_contract


def build_argv(model, *, host="127.0.0.1", port=8000, served_model_name="q35-mx8-study", quality_rows=None):
    from .graph_policy import compilation_settings
    if quality_rows not in (None, 32, 64):
        raise ValueError("Quality rows must be 32 or 64")
    contract = (quality_contract(quality_rows) if quality_rows is not None else
                {"max_model_len": 8192, "max_num_seqs": 128, "max_num_batched_tokens": 2048,
                 "kv_cache_memory_bytes": 19 * 2**30, "compilation_config": compilation_settings()})
    return [
        "vllm", "serve", str(model), "--served-model-name", served_model_name,
        "--host", host, "--port", str(port), "--tokenizer", str(model),
        "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--language-model-only",
        "--max-model-len", str(contract["max_model_len"]), "--max-num-seqs", str(contract["max_num_seqs"]),
        "--max-num-batched-tokens", str(contract["max_num_batched_tokens"]),
        *(["--gpu-memory-utilization", "0.85"] if quality_rows is None else
          ["--enable-chunked-prefill", "--logprobs-mode", "raw_logprobs", "--disable-log-stats"]),
        "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float32",
        "--attention-backend", "FLASHINFER", "--quantization", "compressed-tensors",
        "--linear-backend", "flashinfer_cutlass", "--kv-cache-memory", str(contract["kv_cache_memory_bytes"]),
        "--kv-cache-dtype", "fp8_e4m3", "--seed", "20260907",
        "--compilation-config", json.dumps(contract["compilation_config"], separators=(",", ":")),
    ]


def configure_environment(directory, *, quality_rows=None):
    if quality_rows not in (None, 32, 64):
        raise ValueError("Quality rows must be 32 or 64")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    from .install import load_manifest
    manifest = load_manifest()
    cache_root = directory / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    startup_cache = Path(tempfile.mkdtemp(prefix="startup-", dir=cache_root))
    settings = {
        "VLLM_MACH_PROFILE": NAME, "VLLM_MACH_NATIVE_MXFP8": "0",
        "VLLM_MACH_MXFP8_RUN_DIR": str(directory), "VLLM_PLUGINS": "mach",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "MX8_NATIVE_POLICY": "all",
        # A fresh namespace rebuilds recipe/parallel hooks on every boot.
        # AOT precompilation requires caching within this startup namespace.
        "VLLM_USE_AOT_COMPILE": "1", "VLLM_DISABLE_COMPILE_CACHE": "0",
        "TORCH_COMPILE_FORCE_DISABLE_CACHES": "0",
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "0",
        "VLLM_FLASH_ATTN_VERSION": "2",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_MACH_MXFP8_MODE": "quality" if quality_rows is not None else "production",
        "VLLM_CACHE_ROOT": str(startup_cache / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(startup_cache / "inductor"),
        "TRITON_CACHE_DIR": str(startup_cache / "triton"),
        **manifest["required_disabled_legacy_environment"],
        **manifest["required_disabled_compact_environment"],
    }
    if quality_rows is None:
        os.environ.pop("VLLM_MACH_MXFP8_QUALITY_ROWS", None)
    else:
        settings["VLLM_MACH_MXFP8_QUALITY_ROWS"] = str(quality_rows)
    os.environ.update(settings)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--served-model-name", default="q35-mx8-study")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--quality-rows", choices=(32, 64), type=int,
                        help="Accepted precision comparison: M32/4 GiB or M64/19 GiB, FULL decode and PIECEWISE mixed graphs at accepted small sizes; no PW2048; not production throughput")
    args = parser.parse_args()
    model = args.model.resolve()
    argv = build_argv(model, host=args.host, port=args.port,
                      served_model_name=args.served_model_name, quality_rows=args.quality_rows)
    if args.print_command:
        print(json.dumps(argv, indent=2))
        return
    # Cache/compiler settings must precede validation's torch/safetensors imports.
    settings = configure_environment(args.run_dir, quality_rows=args.quality_rows)
    from .profile import check_environment, check_runtime_sources, run_contract
    from .prepare_model import validate_model
    check_environment()
    source = check_runtime_sources()
    identity = validate_model(model, verify_weights=True)
    receipt = {"profile": NAME, **run_contract(), "argv": argv,
               "environment": settings, "runtime_sources": source, "model": identity}
    (args.run_dir / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = argv
    vllm_main()


if __name__ == "__main__":
    main()
