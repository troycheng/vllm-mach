# SPDX-License-Identifier: Apache-2.0
"""Optional community block-FP8 A/B GEMM routes for vLLM 0.29 workers.

Import and register do not initialize CUDA. Install after worker device setup,
before model construction. Weight updates/offload after compilation are outside
this profile's contract; normal load/reload refreshes layer-owned derived scales.
"""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from typing import Any
import weakref


_OP = None
_INSTALLED = False
_ENABLED = (False, False)
_TARGET = None
_ORIGINAL_PREPARE = None
_ORIGINAL_APPLY = None
_SM120_DEVICES = frozenset()
_COUNTS = Counter()
_LAYERS = weakref.WeakSet()
_BUFFER = "_mach_block_fp8_scale64"
torch: Any = None


def _torch():
    global torch
    if torch is None:
        torch = import_module("torch")
    return torch


def select_route(a, b, sa, sb, sb64, *, n64=True, ordered=True):
    """Select only qualified layouts; called inside the opaque op at runtime."""
    torch = _torch()
    if a.ndim != 2 or b.ndim != 2:
        return "stock"
    m, k = a.shape
    common = (
        k in (4096, 9216) and b.shape == (2560, k)
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and sa.dtype == sb.dtype == torch.float32
        and a.is_cuda and a.device.index in _SM120_DEVICES
        and a.device == b.device == sa.device == sb.device
        and a.stride(1) == b.stride(1) == sb.stride(1) == 1
        and sa.shape == (m, k // 128) and sb.shape == (20, k // 128)
    )
    if not common:
        return "stock"
    if ordered and 1 <= m <= 8:
        return "ordered"
    if (n64 and 16 <= m <= 128 and m % 8 == 0
            and a.is_contiguous() and b.is_contiguous() and sb.is_contiguous()
            and sb64.dtype == torch.float32 and sb64.device == a.device
            and sb64.is_contiguous() and sb64.shape == (40, k // 128)
            and sa.stride() == (1, m)):
        return "n64"
    return "stock"


def _impl(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor,
          sb: torch.Tensor, sb64: torch.Tensor, n64: bool,
          ordered: bool) -> torch.Tensor:
    route = select_route(a, b, sa, sb, sb64, n64=n64, ordered=ordered)
    _COUNTS[route + "_calls"] += 1
    if _torch().cuda.is_current_stream_capturing():
        _COUNTS[route + "_capture_calls"] += 1
    if route == "n64":
        return import_module("vllm_mach.fp8.block_native").gemm(a, b, sa, sb64)
    if route == "ordered":
        m, k = a.shape
        out = torch.empty((m, 2560), device=a.device, dtype=torch.bfloat16)
        raw = torch.empty((k // 128, m, 2560), device=a.device, dtype=torch.float32)
        return import_module("vllm_mach.fp8.ordered").gemm(a, sa, b, sb, out, raw)
    return import_module("vllm._custom_ops").cutlass_scaled_mm(
        a, b.T, out_dtype=torch.bfloat16, scale_a=sa, scale_b=sb.T
    )


def _fake(a, b, sa, sb, sb64, n64, ordered):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=_torch().bfloat16)


def register():
    """Register the CUDA op and symbolic fake without querying CUDA."""
    global _OP
    if _OP is None:
        torch = _torch()
        _OP = torch.library.custom_op(
            "vllm_mach::block_fp8_linear", _impl,
            mutates_args=(), device_types="cuda",
        )
        _OP.register_fake(_fake)


def gemm(a, b, sa, sb, sb64, *, n64=True, ordered=True):
    """Opaque dispatcher with runtime M and independently enabled A/B routes."""
    if _OP is None:
        raise RuntimeError("register block-FP8 before model compilation")
    return _OP(a, b, sa, sb, sb64, n64, ordered)


def verify_runtime(*, n64=True):
    """Check the active worker's device and, for A, the binary/schema."""
    global _SM120_DEVICES
    torch = _torch()
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("the qualified block-FP8 profile requires SM120")
    _SM120_DEVICES = _SM120_DEVICES | frozenset({device})
    if n64:
        return import_module("vllm_mach.fp8.block_native").verify_runtime()
    return {}


def _weight_scale(layer):
    scale = getattr(layer, "weight_scale_inv", None)
    return getattr(layer, "weight_scale", None) if scale is None else scale


def _eligible_weight(weight, scale):
    torch = _torch()
    return (
        weight is not None and scale is not None
        and weight.ndim == 2 and tuple(weight.shape) in ((2560, 4096), (2560, 9216))
        and weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
        and weight.is_cuda and weight.device.index in _SM120_DEVICES
        and scale.device == weight.device
        and scale.shape == (20, weight.shape[1] // 128) and scale.is_contiguous()
    )


def refresh_scales(layer):
    """Call before capture; online/post-capture updates are unsupported by profile."""
    torch = _torch()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("block-FP8 scales must be refreshed before graph capture")
    weight = getattr(layer, "weight", None)
    scale = _weight_scale(layer)
    if not _ENABLED[0] or not _eligible_weight(weight, scale):
        if hasattr(layer, _BUFFER):
            delattr(layer, _BUFFER)
        return
    derived = scale.repeat_interleave(2, dim=0)
    if hasattr(layer, _BUFFER):
        setattr(layer, _BUFFER, derived)
    else:
        layer.register_buffer(_BUFFER, derived, persistent=False)
    _LAYERS.add(layer)
    _COUNTS["scale_refreshes"] += 1


def _prepare(self, layer):
    _ORIGINAL_PREPARE(self, layer)
    if self.config.out_dtype == _torch().bfloat16:
        refresh_scales(layer)
    elif hasattr(layer, _BUFFER):
        delattr(layer, _BUFFER)


def _apply(self, layer, x, bias=None, **kwargs):
    torch = _torch()
    params = self._get_layer_params(layer)
    weight = params.weight
    scale = _weight_scale(layer)
    if (self.config.out_dtype != torch.bfloat16
            or not _eligible_weight(weight, scale)):
        return _ORIGINAL_APPLY(self, layer, x, bias, **kwargs)
    if _ENABLED[0] and not hasattr(layer, _BUFFER):
        raise RuntimeError("qualified block-FP8 layer has no prepared derived scale")
    # Keep the vLLM 0.29 input-quant, BF16 output, bias and reshape boundaries.
    input_2d = x.view(-1, x.shape[-1])
    output_shape = [*x.shape[:-1], weight.shape[0]]
    if self.apply_input_quant:
        q_input, input_scale = self.quant_fp8(
            input_2d, params.input_scale, params.input_scale_ub,
            use_triton=self.use_triton,
        )
    else:
        q_input = input_2d
        input_scale = (params.input_scale if params.input_scale is not None
                       else input_2d.new_empty(1))
    derived = getattr(layer, _BUFFER, scale)
    output = gemm(q_input, weight, input_scale, scale, derived,
                  n64=_ENABLED[0], ordered=_ENABLED[1])
    if bias is not None:
        output = output + bias
    return output.to(dtype=self.config.out_dtype).view(*output_shape)


def install(*, n64=True, ordered=True):
    """Install once after device setup, before vLLM constructs its layers."""
    global _INSTALLED, _ENABLED, _TARGET, _ORIGINAL_PREPARE, _ORIGINAL_APPLY
    if _INSTALLED:
        if (n64, ordered) != _ENABLED:
            raise RuntimeError("uninstall block-FP8 before changing A/B switches")
        return False
    if not n64 and not ordered:
        return False
    verify_runtime(n64=n64)
    register()
    cls = import_module(
        "vllm.model_executor.kernels.linear.scaled_mm.cutlass"
    ).CutlassFp8BlockScaledMMKernel
    _ORIGINAL_PREPARE = cls.process_weights_after_loading
    _ORIGINAL_APPLY = cls.apply_weights
    _ENABLED = (n64, ordered)
    cls.process_weights_after_loading = _prepare
    cls.apply_weights = _apply
    _TARGET = cls
    _INSTALLED = True
    return True


def uninstall():
    """Restore the class; existing captured graphs must be discarded by caller."""
    global _INSTALLED, _ENABLED
    if not _INSTALLED:
        return False
    if _TARGET.process_weights_after_loading is not _prepare or _TARGET.apply_weights is not _apply:
        raise RuntimeError("block-FP8 class hooks changed; refusing unsafe restoration")
    _TARGET.process_weights_after_loading = _ORIGINAL_PREPARE
    _TARGET.apply_weights = _ORIGINAL_APPLY
    _INSTALLED = False
    _ENABLED = (False, False)
    return True


def inspect(worker=None):
    """Count eager/capture invocations, not CUDA Graph replay executions."""
    layers = [layer for layer in _LAYERS if hasattr(layer, _BUFFER)]
    return {
        "installed": _INSTALLED, "registered": _OP is not None,
        "n64_enabled": _ENABLED[0], "ordered_enabled": _ENABLED[1],
        "counts": dict(_COUNTS), "derived_scale_layers": len(layers),
        "derived_scale_bytes": sum(getattr(layer, _BUFFER).numel() * 4
                                   for layer in layers),
        "workspace_policy": "independent invocation/stream/graph allocations",
    }
