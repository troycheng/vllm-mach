# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional exact BF16 prefill codec for the FlashInfer TP2 IPC workspace.

The CUDA extensions are distributed separately as vllm-mach-lossless-prefill.
No Mach Python plugin or model adapter is imported. Decode retains the ordinary
FlashInfer path; graph capture requires a separate opt-in and eager warmup.
"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import Any
from weakref import WeakKeyDictionary

import torch
from vllm import envs
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    get_fi_ar_workspace,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform

logger = init_logger(__name__)
# Exact shapes implemented by the validated long-prefill extension.
_LONG_ROWS = frozenset(
    (
        1024,
        1052,
        3000,
        3001,
        3002,
        3003,
        3012,
        3013,
        3014,
        3015,
        3020,
        3021,
        3022,
        3023,
        3028,
        3029,
        3030,
        3031,
        3093,
        3094,
        3095,
        3185,
        3244,
        3658,
        3842,
        3845,
        3866,
        3869,
        3890,
        3891,
        3893,
        3895,
    )
)
_VERIFIED: WeakKeyDictionary[GemmaRMSNorm, set[int]] = WeakKeyDictionary()
_LOADED_CODECS: dict[str, Any] = {}


def _codec_kind(shape: tuple[int, ...], dtype: torch.dtype, tp_size: int) -> str | None:
    if tp_size != 2 or len(shape) != 2 or shape[1] != 5120 or dtype != torch.bfloat16:
        return None
    if shape[0] == 4096:
        return "direct"
    return "long" if shape[0] in _LONG_ROWS else None


def _validate_workspace(workspace: Any, rank: int, rows: int) -> None:
    required = 84213760 if rows == 4096 else rows * 5120 * 4 + rows * 5120 // 256 * 4
    meta = getattr(workspace, "metadata", {})
    if (
        getattr(workspace, "backend", None) != "trtllm"
        or meta.get("tp_size") != 2
        or meta.get("tp_rank") != rank
        or meta.get("hidden_dim") != 5120
        or meta.get("max_token_num", 0) < rows
        or meta.get("buffer_size", 0) < required
    ):
        raise RuntimeError(
            "SM120 lossless prefill requires a compatible TP2 TRT-LLM IPC "
            f"workspace (rank={rank}, rows={rows}, required_bytes={required})"
        )


@lru_cache(maxsize=2)
def _load_codec(kind: str) -> Any:
    for package, expected in (
        ("flashinfer-python", "0.6.18"),
        ("vllm-mach-lossless-prefill", "0.1.0a4"),
    ):
        try:
            version = importlib.metadata.version(package).split("+", 1)[0]
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"SM120 lossless prefill requires {package}=={expected}"
            ) from exc
        if version != expected:
            raise RuntimeError(
                f"SM120 lossless prefill validated with {package}=={expected}, "
                f"got {version}"
            )
    name = f"mach_lossless_prefill_{kind}_ext"
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"Missing {name}; install the compatible prefill codec wheel"
        )
    torch.ops.load_library(spec.origin)
    run = getattr(torch.ops, f"mach_lossless_prefill_{kind}").run
    _LOADED_CODECS[kind] = run
    return run


def try_lossless_prefill(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
    norm_out: torch.Tensor,
    tp_size: int,
    max_token_num: int,
) -> bool:
    """Write the fused outputs and return True, or leave all inputs untouched.

    Unsupported tensor layouts, shapes and devices fall back. Graph capture
    requires VLLM_SM120_LOSSLESS_PREFILL_GRAPH and prior eager warmup.
    An explicitly enabled codec with missing binaries or incompatible workspace
    metadata fails before touching the input. Verification is diagnostic only.
    """
    kind = _codec_kind(tuple(hidden_states.shape), hidden_states.dtype, tp_size)
    if kind is None or not hidden_states.is_cuda:
        return False
    if not current_platform.is_device_capability(120):
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and not envs.VLLM_SM120_LOSSLESS_PREFILL_GRAPH:
        return False
    if (
        not hidden_states.is_contiguous()
        or residual.shape != hidden_states.shape
        or norm_out.shape != hidden_states.shape
        or tuple(norm.weight.shape) != (5120,)
        or any(
            t.dtype != hidden_states.dtype
            or t.device != hidden_states.device
            or not t.is_contiguous()
            for t in (residual, norm_out, norm.weight)
        )
    ):
        return False

    run: Any
    if capturing:
        run = _LOADED_CODECS.get(kind)
        if run is None:
            raise RuntimeError(
                "Lossless prefill Graph requires eager warmup before capture"
            )
    rows = hidden_states.shape[0]
    rank = get_tensor_model_parallel_rank()
    workspace = get_fi_ar_workspace(
        world_size=tp_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=5120,
        dtype=hidden_states.dtype,
        group=get_tp_group().cpu_group,
    )
    _validate_workspace(workspace, rank, rows)
    if not capturing:
        run = _load_codec(kind)
    verify = (
        not capturing
        and envs.VLLM_SM120_LOSSLESS_PREFILL_VERIFY
        and rows not in _VERIFIED.get(norm, set())
    )
    if verify:
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
            flashinfer_trtllm_fused_allreduce_norm,
        )

        reference = hidden_states.clone()
        reference_norm = torch.empty_like(norm_out)
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=reference,
            residual=residual,
            rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon,
            world_size=tp_size,
            weight_bias=1.0,
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
            pattern_code=1,
            norm_out=reference_norm,
        )
    args = (
        hidden_states,
        residual,
        norm.weight,
        workspace.workspace_tensor,
        hidden_states,
        norm_out,
        rank,
        int(workspace.metadata["buffer_size"]),
        float(norm.variance_epsilon),
        1.0,
    )
    if kind == "direct":
        run(*args, 1, True)
    else:
        run(*args, True)
    if verify:
        if not (
            torch.equal(hidden_states.view(torch.int16), reference.view(torch.int16))
            and torch.equal(
                norm_out.view(torch.int16), reference_norm.view(torch.int16)
            )
        ):
            raise RuntimeError(
                f"Lossless prefill differs from FlashInfer: rank={rank}, M={rows}"
            )
        _VERIFIED.setdefault(norm, set()).add(rows)
        logger.info_once("SM120 lossless prefill verified: rank=%s M=%s", rank, rows)
    logger.info_once(
        "SM120 lossless prefill active: rank=%s M=%s codec=%s capture=%s",
        rank,
        rows,
        kind,
        capturing,
    )
    return True
