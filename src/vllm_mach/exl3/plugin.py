"""vLLM general-plugin entry point.

Registration intentionally imports no CUDA extension. The extension is resolved only
when a dense EXL3 linear is invoked after loading.
"""

from importlib.metadata import version

from vllm.model_executor.layers.quantization import register_quantization_config

from .dense_adapter import Exl3Config


def register() -> None:
    """Register the provider; safe when vLLM invokes plugins more than once."""

    installed = version("vllm").split("+", 1)[0]
    if installed != "0.28.0":
        raise RuntimeError(
            "vLLM Mach 0.1.0 requires vLLM 0.28.0; "
            f"found {installed}."
        )
    register_quantization_config("exl3")(Exl3Config)
