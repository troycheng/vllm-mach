# SPDX-License-Identifier: Apache-2.0
"""Launch the complete, versioned 4B MXFP8 profile from one checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .ssm import DTYPES, ENV as SSM_ENV, ORDERED_BF16_ENV, validate_dtype

from .profile import NAME, quality_contract
from . import kv


def build_argv(model, *, host="127.0.0.1", port=8000, served_model_name="q35-mx8-study", quality_rows=None, ssm_dtype="float32", kv_cache_dtype="fp8_e4m3"):
    validate_dtype(ssm_dtype)
    kv.validate_dtype(kv_cache_dtype)
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
        "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", ssm_dtype,
        "--attention-backend", "FLASHINFER", "--quantization", "compressed-tensors",
        "--linear-backend", "flashinfer_cutlass", "--kv-cache-memory", str(contract["kv_cache_memory_bytes"]),
        "--kv-cache-dtype", kv_cache_dtype, "--seed", "20260907",
        "--compilation-config", json.dumps(contract["compilation_config"], separators=(",", ":")),
    ]


def configure_environment(directory, *, quality_rows=None, hybrid_head=True, ssm_dtype="float32", ordered_bf16_ssm=True, kv_cache_dtype="fp8_e4m3", kv_scales=None):
    validate_dtype(ssm_dtype)
    kv.validate_dtype(kv_cache_dtype)
    if kv_scales is not None:
        if kv_cache_dtype != "fp8_e4m3":
            raise ValueError("Explicit KV scales require fp8_e4m3")
        kv_scales = str(Path(kv_scales).resolve())
        kv.load_profile(kv_scales)
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
        "VLLM_MACH_MXFP8_COMPACT_HEAD": str(int(hybrid_head)),
        "VLLM_MACH_LM_HEAD_SAMPLING": str(int(hybrid_head)),
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
    settings[SSM_ENV] = ssm_dtype
    settings[ORDERED_BF16_ENV] = str(int(ordered_bf16_ssm))
    settings[kv.DTYPE_ENV] = kv_cache_dtype
    if kv_scales is None:
        os.environ.pop(kv.SCALES_ENV, None)
    else:
        settings[kv.SCALES_ENV] = kv_scales
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
    parser.add_argument("--hybrid-head", "--nvfp4-lm-head", "--lm-head-sampling",
                        dest="hybrid_head", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable hybrid LM-head for greedy and eligible top-k/presence decode (default on)")
    parser.add_argument("--quality-rows", choices=(32, 64), type=int,
                        help="Accepted precision comparison: M32/4 GiB or M64/19 GiB, FULL decode and PIECEWISE mixed graphs at accepted small sizes; no PW2048; not production throughput")
    parser.add_argument("--ssm-dtype", "--mamba-ssm-cache-dtype", choices=DTYPES, default="float32",
                        help="SSM state storage; bfloat16 defaults to ordered W4 GDN")
    parser.add_argument("--ordered-bf16-ssm", action=argparse.BooleanOptionalAction, default=True,
                        help="Ordered W4 BF16 state (default on); --no-ordered-bf16-ssm selects stock GDN")
    parser.add_argument("--kv-cache-dtype", choices=kv.DTYPES, default="fp8_e4m3",
                        help="Attention KV storage (default: fp8_e4m3); use bfloat16 for the unquantized KV baseline")
    parser.add_argument("--kv-scales", type=Path,
                        help="Explicit FP8 KV scale profile; default uses model metadata")
    args = parser.parse_args()
    model = args.model.resolve()
    argv = build_argv(model, host=args.host, port=args.port,
                      served_model_name=args.served_model_name, quality_rows=args.quality_rows, ssm_dtype=args.ssm_dtype,
                      kv_cache_dtype=args.kv_cache_dtype)
    if args.print_command:
        print(json.dumps(argv, indent=2))
        return
    # Cache/compiler settings must precede validation's torch/safetensors imports.
    settings = configure_environment(args.run_dir, quality_rows=args.quality_rows, hybrid_head=args.hybrid_head,
                                     ssm_dtype=args.ssm_dtype, ordered_bf16_ssm=args.ordered_bf16_ssm,
                                     kv_cache_dtype=args.kv_cache_dtype, kv_scales=args.kv_scales)
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
