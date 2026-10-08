# SPDX-License-Identifier: Apache-2.0
"""Native SM120 W8A8 adapter using the shared MXFP6 runtime conventions."""
from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.platforms import PlatformEnum
from vllm.utils.torch_utils import direct_register_custom_op

from .dense import is_mxfp6_sm120_available

_REGISTERED = False
_FLASHINFER_BUCKETS = (1, 2, 4, 8, 16, 24, 32)
_B12X_SHAPES = frozenset({
    (12288, 2560), (2560, 4096), (18432, 2560),
    (2560, 9216), (10240, 2560),
})


def _backend():
    backend = os.environ.get("VLLM_MACH_MXFP8_BACKEND", "native")
    if backend not in ("native", "flashinfer"):
        raise ValueError("VLLM_MACH_MXFP8_BACKEND must be native or flashinfer")
    return backend


def _flashinfer_backend(m, n, k):
    # Match vllm-shpgy's Qwen3.5-4B decode dispatch. Larger M uses CUTLASS.
    return "b12x" if 0 < m <= 32 and (n, k) in _B12X_SHAPES else "cutlass"


def _flashinfer_gemm(x, weight, weight_scale, *, tune=False):
    codes, scales = torch.ops.mxfp6.quantize_mxfp8(x)
    a = codes.view(x.shape).view(torch.float8_e4m3fn)
    return _packed_gemm(a, weight, scales, weight_scale, tune=tune)


def _runtime():
    return importlib.import_module("mxfp6.mxfp8")


def _pdl_enabled(m):
    return os.getenv("VLLM_MACH_MXFP8_PDL", "0") == "1" and 0 < m <= 32


@lru_cache(maxsize=1)
def _require_pdl_runtime():
    if (not hasattr(torch.ops.mxfp8_sm120, "pdl_version")
            or torch.ops.mxfp8_sm120.pdl_version() < 3
            or not hasattr(torch.ops.mxfp6, "quantize_mxfp8_pdl")):
        raise RuntimeError(
            "MXFP8 PDL requires both rebuilt native libraries with PDL version 3; "
            "set MXFP6_LIBRARY_PATH and MXFP8_LIBRARY_PATH")


def _packed_gemm(a, weight, scales, weight_scale, *, tune=False):
    if _backend() == "flashinfer":
        import flashinfer

        with flashinfer.autotune(tune, tuning_buckets=_FLASHINFER_BUCKETS):
            return flashinfer.mm_mxfp8(
                a, weight.T, scales, weight_scale, out_dtype=torch.bfloat16,
                backend=_flashinfer_backend(a.shape[0], *weight.shape),
            )
    if _pdl_enabled(a.shape[0]):
        _require_pdl_runtime()
        return torch.ops.mxfp8_sm120.gemm_pdl(a, weight, scales, weight_scale)
    return torch.ops.mxfp8_sm120.gemm(a, weight, scales, weight_scale)


def _gemm(x: torch.Tensor, weight: torch.Tensor,
          weight_scale: torch.Tensor) -> torch.Tensor:
    if _backend() == "flashinfer":
        return _flashinfer_gemm(x, weight, weight_scale)
    if _pdl_enabled(x.shape[0]):
        _require_pdl_runtime()
        return torch.ops.mxfp8_sm120.gemm_from_float_pdl(x, weight, weight_scale)
    return torch.ops.mxfp8_sm120.gemm_from_float(x, weight, weight_scale)


def _fake(x: torch.Tensor, weight: torch.Tensor,
          weight_scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _register_op():
    global _REGISTERED
    if not _REGISTERED:
        direct_register_custom_op(
            op_name="mach_mxfp8_sm120_gemm", op_func=_gemm,
            mutates_args=[], fake_impl=_fake,
        )
        _REGISTERED = True


class Mxfp8Sm120LinearKernel(Mxfp8LinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not is_mxfp6_sm120_available(compute_capability):
            return False, "requires the native MX runtime and SM120"
        try:
            runtime = _runtime()
            runtime.load_library()
            for name in ("gemm_from_float", "begin_workspace_planning",
                         "finalize_workspace_planning", "warmup", "MXFP8Tensor"):
                if not hasattr(runtime, name):
                    return False, f"MXFP8 runtime missing {name}; rebuild mxfp6-sm120"
            if not hasattr(torch.ops.mxfp8_sm120, "gemm_from_float"):
                return False, "MXFP8 native library is too old"
            _register_op()
        except Exception as error:
            return False, f"MXFP8 runtime unavailable: {error}"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig):
        return True, None

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        if weight.ndim != 2 or any(d <= 0 or d % 128 for d in weight.shape):
            raise ValueError("Native MXFP8 requires positive N and K divisible by 128")
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Native MXFP8 requires E4M3 weights")
        if getattr(layer, "params_dtype", torch.bfloat16) != torch.bfloat16:
            raise ValueError("Native MXFP8 requires --dtype bfloat16")
        n, k = weight.shape
        scale = layer.weight_scale.data
        if scale.dtype != torch.uint8 or tuple(scale.shape) != (n, k // 32):
            raise ValueError("Native MXFP8 requires uint8 E8M0 scales shaped [N, K/32]")
        layer.weight = torch.nn.Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            _runtime().pack_scales(scale.contiguous()), requires_grad=False)

    def apply_weights(self, layer, x, bias=None):
        n, k = layer.weight.shape
        if x.dtype != torch.bfloat16 or x.shape[-1] != k:
            raise ValueError("Native MXFP8 requires BF16 activations with matching K")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], n))
        y = torch.ops.vllm.mach_mxfp8_sm120_gemm(
            x.reshape(-1, k).contiguous(), layer.weight, layer.weight_scale,
        ).reshape(*x.shape[:-1], n)
        return y if bias is None else y + bias


def register_dense_mxfp8_kernel():
    from vllm.model_executor.kernels import linear

    kernels = linear._POSSIBLE_MXFP8_KERNELS.get(PlatformEnum.CUDA)
    if not isinstance(kernels, list):
        return False
    while Mxfp8Sm120LinearKernel in kernels:
        kernels.remove(Mxfp8Sm120LinearKernel)
    kernels.insert(0, Mxfp8Sm120LinearKernel)
    return True


@torch.inference_mode()
def warmup_mxfp8(model, token_sizes, *, stream=False):
    """Plan the independent W8A8 pool through the existing runner warmup hooks."""
    global _FLASHINFER_BUCKETS
    problems = {}
    for layer in model.modules():
        kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
        if isinstance(kernel, Mxfp8Sm120LinearKernel):
            problems.setdefault(tuple(layer.weight.shape), layer)
    sizes = sorted({m for m in token_sizes if m > 0}, reverse=True)
    if not problems or not sizes:
        return
    runtime = _runtime()
    device = next(iter(problems.values())).weight.device
    if _backend() == "flashinfer":
        if not stream:
            # Tune both decode and prefill before graph capture. Explicit buckets
            # keep a tactic selected at M=32 from masking an M=1/16 regression.
            _FLASHINFER_BUCKETS = tuple(sorted(set(sizes) | {
                m for m in (1, 2, 4, 8, 16, 24, 32) if m <= max(sizes)
            }))
        for layer in problems.values():
            for m in _FLASHINFER_BUCKETS:
                x = torch.zeros((m, layer.weight.shape[1]), device=device,
                                dtype=torch.bfloat16)
                _flashinfer_gemm(x, layer.weight, layer.weight_scale,
                                tune=not stream)
        torch.cuda.synchronize(device)
        return
    # Max-prefill and graph sizes alone miss intermediate Stream-K layouts.
    # Register their high-water marks before freezing the per-stream arena.
    sizes = sorted(set(sizes) | {
        m for m in (64, 128, 256, 512, 1024, 2048) if m <= max(sizes)
    }, reverse=True)
    if not stream:
        runtime.begin_workspace_planning(device)
    for (n, k), layer in problems.items():
        weight = runtime.MXFP8Tensor(layer.weight, layer.weight_scale, n, k)
        for m in sizes:
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            runtime.warmup(x, weight, out_dtype=torch.bfloat16, iterations=1)
    if not stream:
        runtime.finalize_workspace_planning(device)
    torch.cuda.synchronize(device)
