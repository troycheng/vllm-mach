# SPDX-License-Identifier: Apache-2.0

"""Optional native MXFP6 backends for vLLM Mach.

The package is deliberately import-light. Importing this module does not
import ``mxfp6`` or load a CUDA shared library. The optional module is checked
only when the registration wrapper is called to enable the public
``fp8_e4m3`` spelling; the shared library is loaded only when vLLM asks the
kernel selector about a concrete device.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_DENSE_EXPORTS = {
    "Mxfp6Sm120LinearKernel",
    "is_mxfp6_sm120_available",
    "register_vllm_mxfp8_activation",
}


def _optional_runtime_is_installed() -> bool:
    try:
        return importlib.util.find_spec("mxfp6") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def register_dense_kernel() -> bool:
    """Register the native selector when the optional runtime is installed."""

    if not _optional_runtime_is_installed():
        return False
    dense = importlib.import_module(f"{__name__}.dense")
    return dense.register_dense_kernel()


def __getattr__(name: str) -> Any:
    if name not in _DENSE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    dense = importlib.import_module(f"{__name__}.dense")
    value = getattr(dense, name)
    globals()[name] = value
    return value

__all__ = [
    "Mxfp6Sm120LinearKernel",
    "is_mxfp6_sm120_available",
    "register_dense_kernel",
    "register_vllm_mxfp8_activation",
]
