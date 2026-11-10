# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the independent block-FP8 profile."""
from __future__ import annotations

from vllm_mach.stack import BASE_PACKAGES, require_version

from importlib.metadata import distribution, version
import os
from pathlib import Path

NAME = "qwen35-4b-block-fp8-v1"
FEATURES = ("n64", "ordered", "silu", "fa2")
DEPENDENCIES = dict(BASE_PACKAGES)


def selected_features():
    selected = []
    for name in FEATURES:
        value = os.environ.get(f"VLLM_MACH_FP8_{name.upper()}", "0")
        if value not in ("0", "1"):
            raise RuntimeError(f"FP8 feature {name} must be 0 or 1")
        if value == "1":
            selected.append(name)
    return tuple(selected)


def check_environment():
    if os.environ.get("VLLM_MACH_PROFILE") != NAME:
        raise RuntimeError("Explicit block-FP8 profile selection required")
    if not selected_features():
        raise RuntimeError("Select at least one block-FP8 feature")
    for name in ("VLLM_MACH_NATIVE_MXFP8", "VLLM_HYBRID_MXFP8_LM_HEAD",
                 "VLLM_HYBRID_NVFP4_LM_HEAD", "VLLM_QWEN3_5_FUSED_AR_NORM",
                 "VLLM_VOCAB_PARALLEL_GREEDY"):
        if os.environ.get(name, "0") != "0":
            raise RuntimeError(f"Block-FP8 profile cannot be combined with {name}")
    actual = {key: version(key) for key in DEPENDENCIES}
    for key, expected in DEPENDENCIES.items():
        require_version(key, actual[key], expected)
    return actual


def check_runtime_sources():
    from .install import inspect_sources
    site = Path(distribution("vllm").locate_file(""))
    receipt = inspect_sources(site)
    if receipt["state"] != "installed":
        raise RuntimeError("Run vllm-mach-fp8-install --apply before selecting this profile")
    return receipt
