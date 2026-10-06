# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused BF16 SiLU/block-FP8 quantization for vLLM 0.29.0.

Registration is CPU-safe. Workers preload the native wheel after device setup;
the opaque dispatcher preserves the original op outside the qualified boundary.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import threading
from typing import Any

_ENV = "VLLM_MACH_FP8_SILU"
_OP = "silu_and_mul_per_block_quant"
_SCHEMA = (
    "silu_and_mul_per_block_quant(Tensor(a!) out, Tensor input, "
    "Tensor(b!) scales, int group_size, Tensor? scale_ub, "
    "bool is_scale_transposed) -> ()"
)
_LIBRARIES: list[Any] = []
_LOCK = threading.RLock()
_REGISTERED = False
_INSTALLED = False
_NATIVE_LOADED = False
_NATIVE_IDENTITY: dict[str, Any] = {}
_DEVICES: frozenset[int] = frozenset()
_COUNTS: Counter[str] = Counter()


def enabled() -> bool:
    return os.environ.get(_ENV) == "1"


def _fake(out, input, scales, group_size, scale_ub, is_scale_transposed):
    return None


def _supported(out, input, scales, group_size, scale_ub, transposed) -> bool:
    import torch

    if not (_INSTALLED and enabled() and input.is_cuda):
        return False
    if input.device.index not in _DEVICES:
        return False
    if (input.dtype != torch.bfloat16 or out.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32 or group_size != 128
            or scale_ub is not None):
        return False
    if input.ndim != 2 or out.ndim != 2 or scales.ndim != 2:
        return False
    rows = input.shape[0]
    if (not 1 <= rows <= 2048 or input.shape[1] != 18432
            or out.shape != (rows, 9216) or scales.shape != (rows, 72)):
        return False
    if (input.device != out.device or input.device != scales.device
            or not input.is_contiguous() or not out.is_contiguous()):
        return False
    if transposed:
        return scales.stride() == (1, rows)
    return scales.is_contiguous()


def _dispatch(out, input, scales, group_size, scale_ub, is_scale_transposed):
    import torch

    if not _supported(out, input, scales, group_size, scale_ub,
                      is_scale_transposed):
        _COUNTS["fallback"] += 1
        torch.ops._C.silu_and_mul_per_block_quant(
            out, input, scales, group_size, scale_ub, is_scale_transposed
        )
        return
    capturing = torch.cuda.is_current_stream_capturing()
    torch.ops.mach_fp8_activation.run(out, input, scales, is_scale_transposed)
    _COUNTS["native"] += 1
    _COUNTS["capture_native" if capturing else "eager_native"] += 1


def register():
    """Register the mutation schema/fake contract without initializing CUDA."""
    global _REGISTERED
    import torch

    if _REGISTERED:
        return torch.ops.vllm_mach_fp8.silu_and_mul_per_block_quant.default
    with _LOCK:
        if not _REGISTERED:
            library = torch.library.Library("vllm_mach_fp8", "FRAGMENT")
            library.define(_SCHEMA)
            library.impl(_OP, _dispatch, "CompositeExplicitAutograd")
            torch.library.register_fake(f"vllm_mach_fp8::{_OP}", _fake)
            _LIBRARIES.append(library)
            _REGISTERED = True
    return torch.ops.vllm_mach_fp8.silu_and_mul_per_block_quant.default


def get_fused_op(group_size: int = 128):
    """Return the unchanged vLLM op when the SiLU feature is disabled."""
    import torch

    if group_size != 128 or not enabled():
        return torch.ops._C.silu_and_mul_per_block_quant.default
    return register()


def verify_runtime() -> dict[str, Any]:
    """Load the prebuilt native wheel in a worker, never compile at runtime."""
    global _NATIVE_LOADED, _NATIVE_IDENTITY
    import torch

    with _LOCK:
        if version("vllm") != "0.29.0":
            raise RuntimeError("FP8 activation requires vllm==0.29.0")
        if torch.__version__ != "2.13.0+cu130":
            raise RuntimeError("FP8 activation wheel requires Torch 2.13.0+cu130")
        if torch.version.cuda != "13.0":
            raise RuntimeError("FP8 activation wheel requires Torch CUDA 13.0")
        if not _NATIVE_LOADED:
            spec = importlib.util.find_spec("mach_fp8_activation_ext")
            if spec is None or spec.origin is None:
                raise RuntimeError(
                    "Install the prebuilt vllm-mach-fp8-activation wheel"
                )
            path = Path(spec.origin).resolve()
            torch.ops.load_library(str(path))
            schema = str(torch.ops.mach_fp8_activation.run.default._schema)
            expected = (
                "Tensor(a!) out", "Tensor input", "Tensor(b!) scales",
                "bool is_scale_transposed", "-> ()",
            )
            if not all(fragment in schema for fragment in expected):
                raise RuntimeError(f"FP8 activation native schema mismatch: {schema}")
            _NATIVE_IDENTITY = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
            _NATIVE_LOADED = True
    return dict(_NATIVE_IDENTITY)


def install() -> bool:
    """Preload the SM120 implementation after the worker's device setup."""
    global _INSTALLED, _DEVICES
    if not enabled():
        return False
    import torch

    with _LOCK:
        if _INSTALLED:
            return False
        register()
        verify_runtime()
        device = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise RuntimeError("FP8 activation opt-in requires an SM120 worker")
        _DEVICES = frozenset({device})
        _INSTALLED = True
    return True


def inspect(worker=None) -> dict[str, Any]:
    """Report Python invocation/capture counters, not CUDA graph replay counts."""
    return {
        "enabled": enabled(), "registered": _REGISTERED,
        "installed": _INSTALLED, "devices": sorted(_DEVICES),
        "native_identity": dict(_NATIVE_IDENTITY), "counts": dict(_COUNTS),
    }
