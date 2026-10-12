# SPDX-License-Identifier: Apache-2.0
"""SSM storage selection; FP32 keeps ordered GDN, BF16 uses stock vLLM."""
import os

ENV = "VLLM_MACH_MXFP8_SSM_DTYPE"
DTYPES = ("float32", "bfloat16")


def validate_dtype(dtype):
    if dtype not in DTYPES:
        raise ValueError(f"SSM dtype must be one of {DTYPES}, got {dtype!r}")
    return dtype


def selected_dtype():
    return validate_dtype(os.environ.get(ENV, "float32"))


def select_gdn(ordered):
    if selected_dtype() == "float32":
        return ordered
    from . import stock_gdn
    return stock_gdn
