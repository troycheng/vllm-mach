# SPDX-License-Identifier: Apache-2.0
"""Serve the Qwen3.5-2B champion from one self-contained checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .profile import NAME, quality_contract


def build_argv(model, *, host="127.0.0.1", port=8000, served_model_name="q35-2b-study", quality_rows=None):
    from .graph_policy import CAPTURE_SIZES
    cfg = quality_contract(quality_rows) if quality_rows is not None else {
        "max_model_len": 16384, "max_num_seqs": 160, "max_num_batched_tokens": 2048,
        "kv_cache_memory_bytes": 19 * 2**30,
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE",
                               "cudagraph_capture_sizes": list(CAPTURE_SIZES),
                               "max_cudagraph_capture_size": CAPTURE_SIZES[-1]}}
    return ["vllm", "serve", str(model), "--tokenizer", str(model),
            "--served-model-name", served_model_name, "--host", host, "--port", str(port),
            "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--language-model-only",
            "--max-model-len", str(cfg["max_model_len"]), "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]),
            "--kv-cache-memory", str(cfg["kv_cache_memory_bytes"]), "--kv-cache-dtype", "bfloat16",
            "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float32",
            "--attention-backend", "FLASH_ATTN", "--quantization", "compressed-tensors",
            "--linear-backend", "flashinfer_cutlass", "--seed", "20260907",
            *(["--enable-chunked-prefill", "--logprobs-mode", "raw_logprobs", "--disable-log-stats"]
              if quality_rows is not None else []),
            "--compilation-config", json.dumps(cfg["compilation_config"], separators=(",", ":"))]


def configure_environment(directory, *, quality_rows=None):
    if quality_rows is not None:
        quality_contract(quality_rows)
    from ..install import load_manifest
    manifest = load_manifest()
    directory = Path(directory).resolve()
    cache = directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    startup = Path(tempfile.mkdtemp(prefix="startup-", dir=cache))
    settings = {"OMP_NUM_THREADS": "1", "VLLM_PLUGINS": "mach",
                "VLLM_MACH_PROFILE": NAME, "VLLM_MACH_NATIVE_MXFP8": "0",
                "VLLM_MACH_MXFP8_RUN_DIR": str(directory), "MX8_NATIVE_POLICY": "all",
                "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_FLASH_ATTN_VERSION": "2",
                "VLLM_MACH_2B_PINNED_IDS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "VLLM_USE_AOT_COMPILE": "1", "VLLM_DISABLE_COMPILE_CACHE": "0",
                "TORCH_COMPILE_FORCE_DISABLE_CACHES": "0", "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "0",
                "VLLM_MACH_MXFP8_MODE": "production" if quality_rows is None else "quality",
                "VLLM_CACHE_ROOT": str(startup / "vllm"),
                "TORCHINDUCTOR_CACHE_DIR": str(startup / "inductor"),
                "TRITON_CACHE_DIR": str(startup / "triton"),
                **manifest["required_disabled_legacy_environment"],
                **manifest["required_disabled_compact_environment"]}
    if quality_rows is None:
        os.environ.pop("VLLM_MACH_MXFP8_QUALITY_ROWS", None)
    else:
        settings["VLLM_MACH_MXFP8_QUALITY_ROWS"] = str(quality_rows)
    os.environ.update(settings)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="q35-2b-study")
    parser.add_argument("--quality-rows", type=int, choices=(4, 32, 64))
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    argv = build_argv(model, host=args.host, port=args.port,
                      served_model_name=args.served_model_name, quality_rows=args.quality_rows)
    if args.print_command:
        print(json.dumps(argv, indent=2))
        return
    settings = configure_environment(args.run_dir, quality_rows=args.quality_rows)
    from .profile import check_environment, check_runtime_sources, run_contract
    from .model import validate_model
    check_environment()
    source = check_runtime_sources()
    identity = validate_model(model, verify_weights=True)
    (args.run_dir / "launch.json").write_text(json.dumps({
        "profile": NAME, **run_contract(), "argv": argv, "environment": settings,
        "runtime_sources": source, "model": identity}, indent=2) + "\n")
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = argv
    vllm_main()


if __name__ == "__main__":
    main()
