# SPDX-License-Identifier: Apache-2.0

"""vLLM Mach general-plugin entry point."""

from importlib.metadata import version
from functools import lru_cache
import hashlib
import os
from pathlib import Path

from .mxfp6 import register_dense_kernel


@lru_cache(maxsize=1)
def _integration_hash() -> str:
    # vLLM's AOT lookup precedes tracing: changes to monkeypatched forwards must
    # invalidate the lookup even when their feature flags have not changed.
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in (".py", ".cu") and path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def register_compile_factors() -> None:
    """Keep AOT graphs separate when Mach changes model forwards at warmup."""
    from vllm import envs

    envs.environment_variables["VLLM_MACH_CODE_HASH"] = _integration_hash
    for name, default in {
        "VLLM_MACH_FUSED_GEMMA_NORM": "1",
        "VLLM_MACH_MXFP8_BACKEND": "native",
        "VLLM_MACH_MXFP8_FUSED_MLP": "1",
        "VLLM_MACH_MXFP8_NORM_QUANT": "0",
        "VLLM_MACH_MXFP8_PDL": "0",
        "VLLM_MACH_GDN_TP1": "0",
        "VLLM_MACH_GDN_PERSISTENT": "0",
        "VLLM_MACH_GDN_BA_OVERLAP": "0",
    }.items():
        envs.environment_variables[name] = (
            lambda name=name, default=default: os.environ.get(name, default)
        )


def register() -> None:
    """Register every compatible vLLM Mach backend."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.29.0":
        raise RuntimeError(f"vLLM Mach requires vLLM 0.29.0; found {installed}.")
    register_compile_factors()
    register_dense_kernel()


__all__ = ["register"]
