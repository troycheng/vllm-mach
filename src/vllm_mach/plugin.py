# SPDX-License-Identifier: Apache-2.0

"""vLLM Mach general-plugin entry point."""

from importlib.metadata import version

from vllm.model_executor.layers.quantization import register_quantization_config

from .exl3.dense_adapter import Exl3Config
from .exl3.fused_mlp import install as install_fused_mlp
from .mxfp6 import register_dense_kernel


def register() -> None:
    """Register every compatible vLLM Mach backend."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.28.0":
        raise RuntimeError(
            f"vLLM Mach 0.1.0a1 requires vLLM 0.28.0; found {installed}."
        )
    register_quantization_config("exl3")(Exl3Config)
    register_dense_kernel()
    install_fused_mlp()


__all__ = ["register"]
