# SPDX-License-Identifier: Apache-2.0

"""vLLM Mach general-plugin entry point."""

import os
from importlib.metadata import version

from .mxfp6 import register_dense_kernel


def register() -> None:
    """Register every compatible vLLM Mach backend."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.29.0":
        raise RuntimeError(f"vLLM Mach requires vLLM 0.29.0; found {installed}.")
    profile = os.environ.get("VLLM_MACH_PROFILE")
    if profile == "qwen35-2b-mxfp8-champion-v1":
        from .mxfp8.two_b.profile import install_worker_hook

        install_worker_hook()
        return
    if profile == "qwen35-4b-block-fp8-v1":
        from .fp8.worker import install_worker_hook

        install_worker_hook()
        return
    if profile:
        from .mxfp8.profile import NAME, install_worker_hook

        if profile != NAME:
            raise RuntimeError(f"Unknown vLLM Mach profile: {profile}")
        install_worker_hook()
        return
    register_dense_kernel()
    if os.environ.get("VLLM_MACH_NATIVE_MXFP8") == "1":
        from .mxfp8.worker import install_worker_hook

        install_worker_hook()


__all__ = ["register"]
