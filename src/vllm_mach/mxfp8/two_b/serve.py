# SPDX-License-Identifier: Apache-2.0
"""Serve the Qwen3.5-2B champion from one self-contained checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from ..ssm import DTYPES, ENV as SSM_ENV, ORDERED_BF16_ENV, validate_dtype

from .profile import ATTENTION_BACKENDS, NAME, quality_contract


def build_argv(model, *, host="127.0.0.1", port=8000, served_model_name="q35-2b-study",
               quality_rows=None, attention_backend="FLASH_ATTN", ssm_dtype="float32"):
    validate_dtype(ssm_dtype)
    from .graph_policy import CAPTURE_SIZES
    if attention_backend not in ATTENTION_BACKENDS:
        raise ValueError("Unsupported 2B attention backend")
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
            "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", ssm_dtype,
            "--attention-backend", attention_backend, "--quantization", "compressed-tensors",
            "--linear-backend", "flashinfer_cutlass", "--seed", "20260907",
            *(["--enable-chunked-prefill", "--logprobs-mode", "raw_logprobs", "--disable-log-stats"]
              if quality_rows is not None else []),
            "--compilation-config", json.dumps(cfg["compilation_config"], separators=(",", ":"))]


def configure_environment(directory, *, quality_rows=None, attention_backend="FLASH_ATTN",
                          hybrid_head=False, ssm_dtype="float32", ordered_bf16_ssm=True):
    validate_dtype(ssm_dtype)
    if attention_backend not in ATTENTION_BACKENDS:
        raise ValueError("Unsupported 2B attention backend")
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
                "VLLM_MACH_2B_ATTENTION_BACKEND": attention_backend,
                "VLLM_MACH_2B_HYBRID_HEAD": "1" if hybrid_head else "0",
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
    settings[SSM_ENV] = ssm_dtype
    settings[ORDERED_BF16_ENV] = str(int(ordered_bf16_ssm))
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
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="FLASH_ATTN")
    parser.add_argument("--hybrid-head", action="store_true",
                        help="Use K128 NVFP4 proposals with BF16 refinement for eligible greedy decode")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--ssm-dtype", "--mamba-ssm-cache-dtype", choices=DTYPES, default="float32",
                        help="SSM state storage; bfloat16 defaults to ordered W4 GDN")
    parser.add_argument("--ordered-bf16-ssm", action=argparse.BooleanOptionalAction, default=True,
                        help="Ordered W4 BF16 state (default on); --no-ordered-bf16-ssm selects stock GDN")
    args = parser.parse_args()
    model = args.model.resolve()
    argv = build_argv(model, host=args.host, port=args.port,
                      served_model_name=args.served_model_name, quality_rows=args.quality_rows,
                      attention_backend=args.attention_backend, ssm_dtype=args.ssm_dtype)
    if args.print_command:
        launch = [sys.executable, "-m", __package__ + ".serve",
                  *[arg for arg in sys.argv[1:] if arg != "--print-command"]]
        print(json.dumps({"argv": argv, "launch_command": launch}, indent=2))
        return
    settings = configure_environment(args.run_dir, quality_rows=args.quality_rows,
                                     attention_backend=args.attention_backend, hybrid_head=args.hybrid_head, ssm_dtype=args.ssm_dtype, ordered_bf16_ssm=args.ordered_bf16_ssm)
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
