# SPDX-License-Identifier: Apache-2.0
"""Launch the Qwen3.8-27B native MXFP6 TP2 profile."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def profile_environment(args: argparse.Namespace) -> dict[str, str]:
    # Explicit values prevent stale profile flags from changing this run.
    return {
        "VLLM_PLUGINS": "mach",
        "MXFP6_AUTOTUNE": "off",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_QWEN3_5_FUSED_AR_NORM": "1",
        "VLLM_QWEN3_5_FP16_SSM": str(int(args.fp16_ssm)),
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
        "VLLM_SM120_LOSSLESS_PREFILL": str(int(args.lossless_prefill)),
        "VLLM_SM120_LOSSLESS_PREFILL_GRAPH": "0",
        "VLLM_SM120_LOSSLESS_PREFILL_VERIFY": str(int(args.verify_prefill)),
        "VLLM_SM120_OWNER_PREFILL": str(int(args.owner_prefill)),
        "VLLM_SM120_OWNER_VERIFY": str(int(args.verify_prefill)),
        "VLLM_SM120_OWNER_VERIFY_ONLY_RAGGED": "0",
        "VLLM_SM120_OWNER_MLP_LAYERS": json.dumps(list(range(0, 64, 2))),
        "VLLM_VOCAB_PARALLEL_GREEDY": "1",
        "VLLM_HYBRID_NVFP4_LM_HEAD": str(int(args.nvfp4_lm_head)),
        "VLLM_HYBRID_NVFP4_LM_HEAD_BACKEND": "b12x",
        "VLLM_HYBRID_NVFP4_LM_HEAD_CANDIDATES": "128",
        "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS": "32",
        "VLLM_HYBRID_NVFP4_LM_HEAD_USE_FLASHINFER_TOPK": "1",
    }


def build_command(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], dict]:
    env = dict(os.environ)
    env.update(profile_environment(args))
    config = {
        "mode": "NONE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32],
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--quantization",
        "quark",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-num-seqs",
        "32",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "4096",
        "--no-enable-prefix-caching",
        "--attention-backend",
        "TRITON_ATTN",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--compilation-config",
        json.dumps(config),
    ]
    if args.fp16_ssm:
        command += ["--mamba-ssm-cache-dtype", "float16"]
    return command + extra, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fp16-ssm", action="store_true")
    parser.add_argument("--lossless-prefill", action="store_true")
    parser.add_argument("--owner-prefill", action="store_true")
    parser.add_argument("--nvfp4-lm-head", action="store_true")
    parser.add_argument("--verify-prefill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    command, environment = build_command(args, extra)
    if args.dry_run:
        selected = profile_environment(args)
        print(
            json.dumps(
                {"environment": selected, "command": shlex.join(command)}, indent=2
            )
        )
        return
    from importlib import metadata

    for enabled, package, expected in (
        (args.owner_prefill, "vllm-mach-owner-prefill", "0.1.0a1"),
        (args.lossless_prefill, "vllm-mach-lossless-prefill", "0.1.0a4"),
        (args.nvfp4_lm_head, "b12x", "1.3.0"),
    ):
        if enabled:
            try:
                found = metadata.version(package).split("+", 1)[0]
            except metadata.PackageNotFoundError:
                parser.error(
                    f"Install {package}=={expected} before enabling this option"
                )
            if found != expected:
                parser.error(f"Requires {package}=={expected}; found {found}")

    from .install import check_versions, install_profile

    check_versions()
    result = install_profile(
        Path(metadata.distribution("vllm").locate_file("")),
        Path(metadata.distribution("flashinfer-python").locate_file("")),
    )
    if result["changed_files"]:
        parser.error("Run vllm-mach-install --apply before serving")
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
