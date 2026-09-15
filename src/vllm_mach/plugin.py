# SPDX-License-Identifier: Apache-2.0

"""vLLM Mach general-plugin entry point."""

from importlib.metadata import version

from .mxfp6 import register_dense_kernel


def register() -> None:
    """Register every compatible vLLM Mach backend."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.29.0":
        raise RuntimeError(f"vLLM Mach requires vLLM 0.29.0; found {installed}.")
    register_dense_kernel()


__all__ = ["register"]
