# SPDX-License-Identifier: Apache-2.0

"""Normal champion MLP32 and QKVZ32/64 two-limb activation dispatch.

Install after the native GEMM backend, before loading the model. Opaque op
names and layer ordinals match the measured champion's compiler boundary.
Runtime M branches stay inside those ops; every other M uses native MXFP8.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps
from importlib import import_module
import os
from typing import Any

import torch

from . import native_backend as native

_INSTALLED = False
_ORIGINAL_APPLY: Any = None
_GATEUP_APPLY: Any = None
_QKVZ_APPLY: Any = None
_API: Any = None
_TAGGED_GATEUP_NAMES: tuple[str, ...] = ()
_TAGGED_NAMES: tuple[str, ...] = ()  # GDN ordinal 0..23, not physical layer IDs.
_MLP_COUNTS: Counter = Counter()
_COUNTS: Counter = Counter()
_MLP_CAPTURE_LAYERS: Counter = Counter()
_CAPTURE_LAYERS: dict[int, Counter] = {32: Counter(), 64: Counter()}


def _quant_native(x, weight, scale):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    activation, activation_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    return native.gemm(activation, weight, activation_scale, scale)


def _dual(x, weight, scale):
    hi, s_hi, residual, s_res = _API.quantize(x)
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device,
                      dtype=torch.bfloat16)
    _API.gemm_out(hi, weight, s_hi, scale.view(-1), residual, s_res, out)
    return out


def _gateup_impl(x: torch.Tensor, weight: torch.Tensor,
                 scale: torch.Tensor, layer_id: int) -> torch.Tensor:
    m = int(x.shape[0])
    capture = torch.cuda.is_current_stream_capturing()
    _MLP_COUNTS[("normal_capture" if capture else "normal_eager", m)] += 1
    if m != 32:
        return _quant_native(x, weight, scale)
    if capture:
        _MLP_CAPTURE_LAYERS[layer_id] += 1
    return _dual(x, weight, scale)


def _normal_impl(x: torch.Tensor, weight: torch.Tensor,
                 scale: torch.Tensor, layer_id: int) -> torch.Tensor:
    m = int(x.shape[0])
    capture = torch.cuda.is_current_stream_capturing()
    _COUNTS[("capture" if capture else "eager", m)] += 1
    if m not in (32, 64):
        return _quant_native(x, weight, scale)
    if capture:
        _CAPTURE_LAYERS[m][layer_id] += 1
    # The published API selects QKVZ64 pipereg cfg2 for this exact geometry.
    return _dual(x, weight, scale)


def _fake(x: torch.Tensor, weight: torch.Tensor,
          scale: torch.Tensor, layer_id: int) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device,
                       dtype=torch.bfloat16)


def _eligible(layer, x, n):
    weight, scale = layer.weight, layer.weight_scale
    return (x.dtype == torch.bfloat16 and x.device.type == "cuda"
            and x.ndim == 2 and x.shape[-1] == 2560
            and weight.shape == (n, 2560) and weight.dtype == torch.float8_e4m3fn
            and scale.dtype == torch.uint8
            and weight.device == x.device and scale.device == x.device
            and x.is_contiguous() and weight.is_contiguous() and scale.is_contiguous()
            and scale.numel() == n * 80 and x.device.index in native._SM120_DEVICES)


def install(worker=None) -> bool:
    """Install both chained dispatchers in a TP1 GPU worker after device setup."""
    global _INSTALLED, _ORIGINAL_APPLY, _GATEUP_APPLY, _QKVZ_APPLY, _API
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
    from vllm.utils.torch_utils import direct_register_custom_op

    del worker
    if _INSTALLED:
        return False
    if not native._INSTALLED or native._POLICY != "all":
        raise RuntimeError("champion dual dispatch requires native MXFP8 policy=all")
    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError("champion dual dispatch requires tensor parallel size 1")
    cls = FlashInferCutlassMxfp8LinearKernel
    if cls.apply_weights is not native._apply_weights:
        raise RuntimeError("native MXFP8 must be the immediate MLP32 predecessor")
    native.verify_runtime(require_dual=True)
    _API = import_module("mxfp6.mxfp8_dual")
    # Shared official loader owns ordinary and dual operators in one library.
    library = _API.load_library()
    if str(library.resolve()) != native.inspect()["binary"]["path"]:
        raise RuntimeError("dual and ordinary MXFP8 loaded different native libraries")
    direct_register_custom_op(op_name="mach_mx8_dual_gateup", op_func=_gateup_impl,
                              fake_impl=_fake, mutates_args=[])
    direct_register_custom_op(op_name="mach_mx8_dual_qkvz_both", op_func=_normal_impl,
                              fake_impl=_fake, mutates_args=[])
    _ORIGINAL_APPLY = cls.apply_weights

    @wraps(_ORIGINAL_APPLY)
    def gateup_apply(self, layer, x, bias=None):
        if not getattr(layer, "_mx8_dual_gateup", False) or not _eligible(layer, x, 18432):
            return _ORIGINAL_APPLY(self, layer, x, bias)
        out = torch.ops.vllm.mach_mx8_dual_gateup(
            x, layer.weight, layer.weight_scale, layer._mx8_dual_gateup_id)
        return out if bias is None else out + bias

    _GATEUP_APPLY = gateup_apply

    @wraps(gateup_apply)
    def qkvz_apply(self, layer, x, bias=None):
        if not getattr(layer, "_mx8_dual_qkvz_both", False):
            return gateup_apply(self, layer, x, bias)
        if not _eligible(layer, x, 12288):
            return gateup_apply(self, layer, x, bias)
        out = torch.ops.vllm.mach_mx8_dual_qkvz_both(
            x, layer.weight, layer.weight_scale, layer._mx8_dual_qkvz_both_id)
        return out if bias is None else out + bias

    _QKVZ_APPLY = qkvz_apply
    cls.apply_weights = qkvz_apply
    _INSTALLED = True
    return True


def _kernel(layer):
    method = getattr(layer, "quant_method", None)
    scheme = getattr(layer, "scheme", None) or getattr(method, "scheme", None)
    return getattr(scheme, "kernel", None) or getattr(method, "kernel", None)


def _check_layer(layer, n, cls, name):
    if (not isinstance(_kernel(layer), cls)
            or tuple(layer.weight.shape) != (n, 2560)
            or layer.weight.dtype != torch.float8_e4m3fn
            or layer.weight_scale.dtype != torch.uint8
            or tuple(layer.weight_scale.shape) not in ((n * 80,), (n, 80))):
        raise RuntimeError(f"champion dual backend/layout guard failed: {name}")


def tag_model(worker) -> dict[str, tuple[str, ...]]:
    """Tag exactly 32 MLP gate/up and 24 GDN QKVZ projections after load."""
    global _TAGGED_GATEUP_NAMES, _TAGGED_NAMES
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel as cls
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    if not _INSTALLED:
        raise RuntimeError("install champion dual dispatch before tagging the model")
    mlp, qkvz = [], []
    for name, module in worker.model_runner.model.named_modules():
        if isinstance(module, Qwen2MoeMLP) and module.expert_gate is None:
            layer = module.gate_up_proj
            _check_layer(layer, 18432, cls, name)
            layer._mx8_dual_gateup = True
            layer._mx8_dual_gateup_id = len(mlp)
            mlp.append(name + ".gate_up_proj")
        if isinstance(module, QwenGatedDeltaNetAttention):
            layer = module.in_proj_qkvz
            _check_layer(layer, 12288, cls, name)
            if getattr(layer, "_mx8_dual_qkvz", False) or getattr(layer, "_mx8_dual_qkvz64", False):
                raise RuntimeError("standalone QKVZ dispatcher conflicts with champion dual")
            layer._mx8_dual_qkvz_both = True
            layer._mx8_dual_qkvz_both_id = len(qkvz)
            qkvz.append(name + ".in_proj_qkvz")
    if len(set(mlp)) != 32 or len(mlp) != 32 or len(set(qkvz)) != 24 or len(qkvz) != 24:
        raise RuntimeError(f"champion dual requires 32 MLP and 24 QKVZ layers; got {len(mlp)}/{len(qkvz)}")
    _TAGGED_GATEUP_NAMES, _TAGGED_NAMES = tuple(mlp), tuple(qkvz)
    return {"gateup": _TAGGED_GATEUP_NAMES, "qkvz": _TAGGED_NAMES}


def prepare_model(worker) -> dict[str, tuple[str, ...]]:
    """Tag and eager-launch all three dual geometries before model profiling."""
    tagged = tag_model(worker)
    layers = dict(worker.model_runner.model.named_modules())
    for name, rows in ((tagged["gateup"][0], (32,)), (tagged["qkvz"][0], (32, 64))):
        layer = layers[name]
        for m in rows:
            _kernel(layer).apply_weights(
                layer, torch.zeros((m, 2560), device=worker.device, dtype=torch.bfloat16))
    torch.cuda.synchronize(worker.device)
    return tagged


def inspect_worker(worker=None) -> dict[str, Any]:
    del worker
    return {"mode": "normal", "pid": os.getpid(), "installed": _INSTALLED,
            "tagged_gateup_layers": list(_TAGGED_GATEUP_NAMES),
            "tagged_qkvz_layers": list(_TAGGED_NAMES),
            "gateup_layer_count": len(_TAGGED_GATEUP_NAMES),
            "qkvz_layer_count": len(_TAGGED_NAMES),
            "mlp_python_counts": {f"{name}:M{m}": count for (name, m), count in _MLP_COUNTS.items()},
            "python_counts": {f"{name}:M{m}": count for (name, m), count in _COUNTS.items()},
            "mlp_capture_layers": dict(_MLP_CAPTURE_LAYERS),
            "capture_layers": {str(m): dict(counts) for m, counts in _CAPTURE_LAYERS.items()},
            "m64_variant": "qkvz64_pipereg_cfg2",
            "counter_semantics": "Python eager/capture calls, excludes graph replay"}


def verify_capture(worker=None) -> dict[str, Any]:
    missing = {"mlp32": sorted(set(range(32)) - set(_MLP_CAPTURE_LAYERS))}
    missing.update({f"qkvz{m}": sorted(set(range(24)) - set(counts))
                    for m, counts in _CAPTURE_LAYERS.items()})
    if not _INSTALLED or any(missing.values()):
        raise RuntimeError(f"champion dual graph-capture layer coverage missing: {missing}")
    return inspect_worker(worker)


__all__ = ["install", "tag_model", "prepare_model", "inspect_worker", "verify_capture"]
