# SPDX-License-Identifier: Apache-2.0

"""Workspace and CUDA Graph stream warmup for the SM120 Dense backend."""

from __future__ import annotations

import torch

from .dense import Mxfp6Sm120LinearKernel, _import_mxfp6


def _collect_w6a8_problems(
    model: torch.nn.Module,
) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
    problems: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in model.modules():
        scheme = getattr(layer, "scheme", None)
        kernel = getattr(scheme, "ocp_mx_linear", None)
        if not isinstance(kernel, Mxfp6Sm120LinearKernel):
            continue
        output_features, packed_features = map(int, layer.weight.shape)
        input_features = packed_features * 4 // 3
        problems.setdefault(
            (output_features, input_features),
            (layer.weight, layer.weight_scale),
        )
    return [
        (output_features, input_features, weight, weight_scale)
        for (output_features, input_features), (
            weight,
            weight_scale,
        ) in problems.items()
    ]


def _normalize_sizes(token_sizes: list[int], *, reverse: bool) -> list[int]:
    return sorted({size for size in token_sizes if size > 0}, reverse=reverse)


def _normalize_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return torch.bfloat16


def _warm_w6a8_problems(
    problems: list[tuple[int, int, torch.Tensor, torch.Tensor]],
    token_sizes: list[int],
    dtype: torch.dtype,
    *,
    stop_after_new_lane: int | None = None,
) -> None:
    mxfp6 = _import_mxfp6()
    for output_features, input_features, weight, weight_scale in problems:
        packed_weight = mxfp6.PackedMXFP6Tensor(
            values=weight,
            scales=weight_scale,
            rows=output_features,
            k=input_features,
        )
        for num_tokens in token_sizes:
            x = torch.empty(
                (num_tokens, input_features),
                device=weight.device,
                dtype=dtype,
            ).uniform_(-1.0, 1.0)
            mxfp6.warmup_w6a8(x, packed_weight, out_dtype=dtype, iterations=1)
            if (
                stop_after_new_lane is not None
                and mxfp6.workspace_stats(weight.device)["lanes"] > stop_after_new_lane
            ):
                return


@torch.inference_mode()
def warmup_mxfp6_sm120(
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Autotune W6A8 shapes and freeze the workspace before graph capture."""

    problems = _collect_w6a8_problems(model)
    sizes = _normalize_sizes(token_sizes, reverse=True)
    if not problems or not sizes:
        return

    mxfp6 = _import_mxfp6()
    device = problems[0][2].device
    mxfp6.begin_workspace_planning(device)
    _warm_w6a8_problems(problems, sizes, _normalize_dtype(dtype))
    mxfp6.finalize_workspace_planning(device)
    torch.cuda.synchronize(device)


@torch.inference_mode()
def warmup_mxfp6_sm120_stream(
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Register the active graph-capture stream with the frozen workspace."""

    problems = _collect_w6a8_problems(model)
    sizes = _normalize_sizes(token_sizes, reverse=False)
    if not problems or not sizes:
        return

    mxfp6 = _import_mxfp6()
    device = problems[0][2].device
    lane_count = mxfp6.workspace_stats(device)["lanes"]
    if lane_count == 0:
        return
    _warm_w6a8_problems(
        problems,
        sizes,
        _normalize_dtype(dtype),
        stop_after_new_lane=lane_count,
    )
    torch.cuda.synchronize(device)


__all__ = ["warmup_mxfp6_sm120", "warmup_mxfp6_sm120_stream"]
