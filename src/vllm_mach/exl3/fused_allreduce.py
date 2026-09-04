# SPDX-License-Identifier: Apache-2.0

"""FlashInfer AllReduce, GemmaRMSNorm, and packed MXFP8 bridge."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from . import mxfp6_hybrid

FUSED_AR_ENV = "VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8"
_SUPPORTED_FLASHINFER_HEADER_SHA256 = frozenset(
    {"8e3f0d82c307da6d0b7be769cb672164c14bd8594eb5dc8dbad8fb2091b331df"}
)
_SEEN = False

logger = logging.getLogger(__name__)


def enabled() -> bool:
    raw = os.environ.get(FUSED_AR_ENV, "0").strip().lower()
    if raw not in ("0", "1"):
        raise ValueError(f"{FUSED_AR_ENV} accepts only '0' or '1'; got {raw!r}")
    return raw == "1" and mxfp6_hybrid.active_profile() is not None


def _fallback(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )

    return fused_allreduce_gemma_rms_norm(hidden_states, residual, norm)


def _eligible(hidden_states: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        enabled()
        and hidden_states.is_cuda
        and hidden_states.ndim == 2
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and residual.is_cuda
        and residual.shape == hidden_states.shape
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and int(hidden_states.shape[1]) == 5120
        and 0 < int(hidden_states.shape[0]) <= 32
    )


def _header_path(flashinfer: Any) -> Path:
    return (
        Path(flashinfer.__file__).resolve().parent
        / "data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh"
    )


def _verify_flashinfer_header(flashinfer: Any) -> None:
    path = _header_path(flashinfer)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(
            f"{FUSED_AR_ENV}=1 requires the vLLM Mach FlashInfer profile; "
            f"header not readable at {path}"
        ) from exc
    if digest not in _SUPPORTED_FLASHINFER_HEADER_SHA256:
        raise RuntimeError(
            f"{FUSED_AR_ENV}=1 requires the vLLM Mach FlashInfer profile; "
            f"unsupported trtllm_allreduce_fusion.cuh SHA256 {digest}"
        )


def _load_dependencies() -> SimpleNamespace:
    flashinfer = importlib.import_module("flashinfer")
    _verify_flashinfer_header(flashinfer)
    comm = importlib.import_module("flashinfer.comm")
    runtime = importlib.import_module("mxfp6")
    fi_workspace = importlib.import_module(
        "vllm.distributed.device_communicators.flashinfer_all_reduce"
    )
    parallel = importlib.import_module("vllm.distributed.parallel_state")
    required_comm = (
        "AllReduceFusionPattern",
        "QuantizationSFLayout",
        "allreduce_fusion",
    )
    missing = [name for name in required_comm if not hasattr(comm, name)]
    if missing:
        raise RuntimeError(
            "FlashInfer is missing the fused AR/MXFP8 API: " + ", ".join(missing)
        )
    if not hasattr(runtime, "MXFP8Tensor"):
        raise RuntimeError("mxfp6-sm120 is missing MXFP8Tensor")
    return SimpleNamespace(
        comm=comm,
        runtime=runtime,
        get_workspace=fi_workspace.get_fi_ar_workspace,
        get_rank=parallel.get_tensor_model_parallel_rank,
        get_world_size=parallel.get_tensor_model_parallel_world_size,
        get_tp_group=parallel.get_tp_group,
    )


def fused_allreduce_gemma_rms_norm_mxfp8(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[Any, torch.Tensor]:
    """Fuse TP2 reduction and Gemma RMSNorm into the next MXFP6 MLP input."""

    if not _eligible(hidden_states, residual):
        return _fallback(hidden_states, residual, norm)

    dependencies = _load_dependencies()
    tp_size = int(dependencies.get_world_size())
    if tp_size != 2:
        return _fallback(hidden_states, residual, norm)

    rows, hidden = map(int, hidden_states.shape)
    workspace = dependencies.get_workspace(
        world_size=tp_size,
        rank=dependencies.get_rank(),
        max_token_num=32,
        hidden_dim=hidden,
        dtype=hidden_states.dtype,
        group=dependencies.get_tp_group().cpu_group,
    )
    if (
        workspace is None
        or workspace.backend != "trtllm"
        or not workspace.is_buffer_size_sufficient(
            tp_size=tp_size,
            num_tokens=rows,
            hidden_dim=hidden,
            dtype=hidden_states.dtype,
        )
    ):
        return _fallback(hidden_states, residual, norm)

    groups = hidden // 32
    padded_rows = ((rows + 127) // 128) * 128
    packed_groups = ((groups + 3) // 4) * 4
    scales = torch.empty(
        padded_rows * packed_groups,
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    aligned_rows = ((rows + 3) // 4) * 4
    scale_view = torch.as_strided(
        scales.view(torch.int32),
        (rows, groups // 4),
        (1, aligned_rows),
    )
    values = torch.empty(
        (rows, hidden),
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )
    dependencies.comm.allreduce_fusion(
        input=hidden_states,
        workspace=workspace,
        pattern=(
            dependencies.comm.AllReduceFusionPattern
            .kARResidualRMSNormPerTokenGroupFP8PackedQuant
        ),
        launch_with_pdl=True,
        trigger_completion_at_end=True,
        output=None,
        residual_out=hidden_states,
        norm_out=None,
        quant_out=values,
        scale_out=scale_view,
        residual_in=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.variance_epsilon,
        layout_code=dependencies.comm.QuantizationSFLayout.SWIZZLED_128x4,
        use_oneshot=None,
        fp32_acc=True,
        block_quant_group_size=32,
        weight_bias=1.0,
    )

    global _SEEN
    if not _SEEN:
        _SEEN = True
        logger.warning(
            "vLLM Mach Hybrid is using fused FlashInfer "
            "AllReduce/GemmaRMSNorm/MXFP8 quantization"
        )
    return (
        dependencies.runtime.MXFP8Tensor(
            values.view(torch.uint8), scales, rows, hidden
        ),
        hidden_states,
    )


__all__ = ["FUSED_AR_ENV", "enabled", "fused_allreduce_gemma_rms_norm_mxfp8"]
