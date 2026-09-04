# SPDX-License-Identifier: Apache-2.0

"""Fused activation-quantization bridge for the Qwen Dense Hybrid profile."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from . import mxfp6_hybrid

_FUSED_ENV = "VLLM_MACH_EXL3_MXFP6_FUSED_MLP"
_INSTALLED_ATTRIBUTE = "_vllm_mach_exl3_mxfp6_fused_mlp"
_NOT_APPLICABLE = object()
_SEEN = False

logger = logging.getLogger(__name__)


def enabled() -> bool:
    raw = os.environ.get(_FUSED_ENV, "1").strip().lower()
    if raw not in ("0", "1"):
        raise ValueError(f"{_FUSED_ENV} accepts only '0' or '1'; got {raw!r}")
    return raw == "1" and mxfp6_hybrid.active_profile() is not None


def _single_down_weight(layer: Any, rows: int) -> Any | None:
    state = mxfp6_hybrid.state_for_rows(layer, rows)
    if state is None or state.merged_weight is not None or len(state.weights) != 1:
        return None
    return next(iter(state.weights.values()))


def _try_fused_forward(module: Any, x: torch.Tensor) -> Any:
    if not enabled() or x.ndim != 2 or x.dtype != torch.bfloat16:
        return _NOT_APPLICABLE

    gate_up_layer = module.gate_up_proj
    down_layer = module.down_proj
    gate_up_state = mxfp6_hybrid.state_for_rows(gate_up_layer, int(x.shape[0]))
    down_weight = _single_down_weight(down_layer, int(x.shape[0]))
    eligible = (
        gate_up_state is not None
        and gate_up_state.merged_weight is not None
        and down_weight is not None
        and getattr(gate_up_layer, "bias", None) is None
        and getattr(down_layer, "bias", None) is None
        and getattr(module, "expert_gate", None) is None
        and bool(getattr(down_layer, "input_is_parallel", False))
    )
    if not eligible:
        return _NOT_APPLICABLE

    gate_up_result = gate_up_layer(x)
    gate_up = gate_up_result[0] if isinstance(gate_up_result, tuple) else gate_up_result
    activation = mxfp6_hybrid.fused_silu_mxfp8(gate_up)
    output = mxfp6_hybrid.apply_mxfp8_weight(activation, down_weight)
    if bool(getattr(down_layer, "reduce_results", False)) and int(
        getattr(down_layer, "tp_size", 1)
    ) > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce

        output = tensor_model_parallel_all_reduce(output)

    global _SEEN
    if not _SEEN:
        _SEEN = True
        logger.warning(
            "vLLM Mach Hybrid is using fused SiLU/mul and MXFP8 activation "
            "quantization for Dense MLPs"
        )
    return output


def install() -> bool:
    """Install the narrow Qwen MLP hook when the Hybrid profile is active."""

    if not enabled():
        return False
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    if getattr(Qwen2MoeMLP, _INSTALLED_ATTRIBUTE, False):
        return False
    reference_forward = Qwen2MoeMLP.forward

    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        result = _try_fused_forward(self, x)
        if result is _NOT_APPLICABLE:
            return reference_forward(self, x)
        return result

    Qwen2MoeMLP.forward = forward
    setattr(Qwen2MoeMLP, _INSTALLED_ATTRIBUTE, True)
    return True


__all__ = ["enabled", "install"]
