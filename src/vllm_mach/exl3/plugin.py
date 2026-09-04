"""vLLM Mach general-plugin entry point."""

from importlib.metadata import version

from vllm.model_executor.layers.quantization import register_quantization_config

from ..mxfp6 import register_dense_kernel
from .dense_adapter import Exl3Config


def register() -> None:
    """Register the provider; safe when vLLM invokes plugins more than once."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.28.0":
        raise RuntimeError(
            "vLLM Mach 0.1.0a1 requires vLLM 0.28.0; "
            f"found {installed}."
        )
    register_quantization_config("exl3")(Exl3Config)
    register_dense_kernel()
