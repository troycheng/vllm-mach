# SPDX-License-Identifier: Apache-2.0
"""SSM storage selection, with default ordered BF16 state replay."""
import os

ENV = "VLLM_MACH_MXFP8_SSM_DTYPE"
ORDERED_BF16_ENV = "VLLM_MACH_MXFP8_ORDERED_BF16_SSM"
DTYPES = ("float32", "bfloat16")


def validate_dtype(dtype):
    if dtype not in DTYPES:
        raise ValueError(f"SSM dtype must be one of {DTYPES}, got {dtype!r}")
    return dtype


def selected_dtype():
    return validate_dtype(os.environ.get(ENV, "float32"))


def ordered_enabled():
    flag = os.environ.get(ORDERED_BF16_ENV, "1")
    if flag not in ("0", "1"):
        raise ValueError(f"{ORDERED_BF16_ENV} must be 0 or 1")
    return selected_dtype() == "float32" or flag == "1"


def select_gdn(ordered):
    if ordered_enabled():
        return ordered
    from . import stock_gdn
    return stock_gdn
