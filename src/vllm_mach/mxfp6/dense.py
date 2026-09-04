# SPDX-License-Identifier: Apache-2.0

"""Native SM120 Dense MXFP6 backend for vLLM 0.28.

This is framework glue only.  The CUDA operators and checkpoint format remain
owned by the optional ``mxfp6-sm120`` package.  The implementation follows the
public vLLM 0.28 kernel contract and keeps vLLM's emulation kernel as the next
selector candidate when the optional package is unavailable.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import torch
from vllm.model_executor.kernels.linear.mxfp6.base import (
    MxFp6LinearKernel,
    MxFp6LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_REQUIRED_API = (
    "PackedMXFP6Tensor",
    "gemm_from_float",
    "is_available",
    "load_library",
    "pack_scales",
    "warmup_w6a8",
    "begin_workspace_planning",
    "finalize_workspace_planning",
    "workspace_stats",
)
_CUSTOM_OP_REGISTERED = False


def _import_mxfp6() -> ModuleType:
    return importlib.import_module("mxfp6")


def _compute_capability_number(compute_capability: Any) -> int | None:
    if compute_capability is None:
        return None
    if isinstance(compute_capability, tuple):
        if len(compute_capability) != 2:
            return None
        return int(compute_capability[0]) * 10 + int(compute_capability[1])
    return int(compute_capability)


def _has_required_api(mxfp6: ModuleType) -> bool:
    return all(hasattr(mxfp6, name) for name in _REQUIRED_API)


def is_mxfp6_sm120_available(
    compute_capability: int | tuple[int, int] | None = None,
) -> bool:
    """Return whether the optional package can execute on the current device."""

    capability = _compute_capability_number(compute_capability)
    if capability is None:
        try:
            device_capability = current_platform.get_device_capability()
        except Exception:  # noqa: BLE001 - optional backend must fail closed
            return False
        capability = _compute_capability_number(device_capability)
    if capability != 120:
        return False

    try:
        mxfp6 = _import_mxfp6()
        if not _has_required_api(mxfp6):
            return False
        mxfp6.load_library()
        available = bool(mxfp6.is_available())
        if available:
            _ensure_custom_op_registered()
        return available
    except Exception:  # noqa: BLE001 - optional backend must fail closed
        # Kernel selection must remain safe when the optional wheel is absent,
        # has an ABI mismatch, or is being inspected on a CPU-only controller.
        return False


def _mxfp6_sm120_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    mxfp6 = _import_mxfp6()
    mxfp6.load_library()
    return torch.ops.mxfp6.gemm_from_float(
        x,
        weight,
        weight_scale,
        output_features,
        1.0,
        output_dtype,
    )


def _mxfp6_sm120_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    del weight, weight_scale
    return torch.empty(
        (x.shape[0], output_features),
        device=x.device,
        dtype=output_dtype,
    )


def _ensure_custom_op_registered() -> None:
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return
    direct_register_custom_op(
        op_name="mach_mxfp6_sm120_gemm",
        op_func=_mxfp6_sm120_gemm_impl,
        mutates_args=[],
        fake_impl=_mxfp6_sm120_gemm_fake,
    )
    _CUSTOM_OP_REGISTERED = True


class Mxfp6Sm120LinearKernel(MxFp6LinearKernel):
    """Native SM120 W6A8 backend for vLLM's MXFP6 linear interface."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not is_mxfp6_sm120_available(compute_capability):
            return False, "mxfp6-sm120 is unavailable or requires an SM120 GPU"
        return True, None

    @classmethod
    def can_implement(
        cls, config: MxFp6LinearLayerConfig
    ) -> tuple[bool, str | None]:
        if config.weight_quant_key != kMxfp6E3M2Static:
            return False, "requires static MXFP6 E3M2 weights"
        if config.activation_quant_key != kMxfp8Dynamic:
            return False, "requires dynamic MXFP8 E4M3 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        mxfp6 = _import_mxfp6()
        layer.weight = torch.nn.Parameter(
            layer.weight.data.contiguous(), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            mxfp6.pack_scales(layer.weight_scale.data.contiguous()),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_features = int(layer.weight.shape[0])
        input_features = int(x.shape[-1])
        if output_features % 8 != 0 or input_features % 128 != 0:
            raise ValueError(
                "mxfp6-sm120 requires output features divisible by 8 and "
                f"input features divisible by 128; got N={output_features}, "
                f"K={input_features}"
            )
        _ensure_custom_op_registered()
        output_shape = (*x.shape[:-1], output_features)
        y = torch.ops.vllm.mach_mxfp6_sm120_gemm(
            x.reshape(-1, input_features).contiguous(),
            layer.weight,
            layer.weight_scale,
            output_features,
            x.dtype,
        ).reshape(output_shape)
        if bias is not None:
            y = y + bias
        return y


def register_vllm_mxfp8_activation() -> bool:
    """Teach vLLM 0.28's Quark OCP-MX scheme about ``mxfp8_e4m3``.

    The public MXFP6 SM120 checkpoint records dynamic E4M3 activations as
    ``fp8_e4m3``. Quark normalizes that metadata to ``mxfp8_e4m3`` before
    looking up the activation key. vLLM 0.28 has the corresponding
    ``kMxfp8Dynamic`` key but does not map this spelling in ``QuarkOCP_MX``.
    Keep the compatibility patch conditional on the optional wheel so a stock
    installation retains its original fail-closed behavior.
    """

    try:
        mxfp6 = _import_mxfp6()
        if not _has_required_api(mxfp6):
            return False
        quark_ocp_mx = importlib.import_module(
            "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx"
        )
        from vllm.model_executor.layers.quantization.utils import quant_utils

        key = getattr(quant_utils, "kMxfp8Dynamic", None)
        activation_map = getattr(quark_ocp_mx, "_ACTIVATION_QUANT_KEY_MAP", None)
        if key is None or not isinstance(activation_map, dict):
            return False
        activation_map.setdefault("mxfp8_e4m3", key)
        return activation_map.get("mxfp8_e4m3") == key
    except Exception:  # noqa: BLE001 - optional bridge must not block EXL3
        # This is an optional compatibility bridge.  A missing mxfp6 wheel,
        # an older vLLM layout, or a CPU-only controller must not prevent the
        # EXL3 provider from registering.
        return False


def register_dense_kernel() -> bool:
    """Prepend the optional kernel to vLLM's CUDA MXFP6 selector.

    vLLM 0.28's registration helper appends kernels, which would leave the
    always-available emulation backend ahead of this native implementation.
    This pinned compatibility bridge updates the private selector list so the
    native class gets first refusal; all other quantization schemes and the
    emulation fallback remain untouched.
    """

    registered = False
    try:
        from vllm.model_executor.kernels import linear
        registry = getattr(linear, "_POSSIBLE_MXFP6_KERNELS", None)
        if isinstance(registry, dict):
            cuda_kernels = registry.get(PlatformEnum.CUDA)
            if isinstance(cuda_kernels, list):
                while Mxfp6Sm120LinearKernel in cuda_kernels:
                    cuda_kernels.remove(Mxfp6Sm120LinearKernel)
                cuda_kernels.insert(0, Mxfp6Sm120LinearKernel)
                registered = True
    except Exception:  # noqa: BLE001 - private vLLM bridge is best effort
        # Keep plugin discovery usable when this private vLLM 0.28 bridge is
        # unavailable; the normal emulation path remains owned by vLLM.
        registered = False

    # The activation spelling is needed before a Quark OCP-MX layer constructs
    # its linear kernel.  It is safe to attempt this even when the selector
    # registry was unavailable because both operations are independent.
    register_vllm_mxfp8_activation()
    return registered


__all__ = [
    "Mxfp6Sm120LinearKernel",
    "is_mxfp6_sm120_available",
    "register_dense_kernel",
    "register_vllm_mxfp8_activation",
]
