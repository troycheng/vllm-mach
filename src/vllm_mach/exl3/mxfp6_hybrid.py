# SPDX-License-Identifier: Apache-2.0

"""Selective EXL3-to-MXFP6 runtime conversion for validated Dense profiles."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

PROFILE_ENV = "VLLM_MACH_EXL3_MXFP6_PROFILE"
QWEN38_27B_PROFILE = "qwen38-27b"
_SUPPORTED_RUNTIME_VERSION = "0.2.1"
_SUPPORTED_W6A8_ABI = "native-w6a8-30-v5"
_STATE_ATTRIBUTE = "_mach_exl3_mxfp6"


class HybridRoute(Enum):
    ALL_ROWS = "all-rows"
    PREFILL_ONLY = "prefill-only"


@dataclass(frozen=True)
class HybridPackedWeight:
    values: torch.Tensor
    scales: torch.Tensor
    rows: int
    k: int

    @property
    def device(self) -> torch.device:
        return self.values.device


@dataclass(frozen=True)
class HybridState:
    route: HybridRoute
    weights: dict[Any, HybridPackedWeight]
    merged_weight: HybridPackedWeight | None = None
    prefill_min_rows: int = 128

    def active_for_rows(self, rows: int) -> bool:
        return self.route is HybridRoute.ALL_ROWS or rows >= self.prefill_min_rows


_ALL_ROWS_SUFFIXES = (
    ".mlp.gate_up_proj",
    ".mlp.down_proj",
    ".linear_attn.out_proj",
    ".self_attn.o_proj",
)
_PREFILL_ONLY_SUFFIXES = (
    ".linear_attn.in_proj_qkvz",
    ".self_attn.qkv_proj",
)


def active_profile() -> str | None:
    raw = os.environ.get(PROFILE_ENV, "").strip().lower()
    if not raw:
        return None
    if raw != QWEN38_27B_PROFILE:
        raise ValueError(
            f"{PROFILE_ENV} only accepts {QWEN38_27B_PROFILE!r}; got {raw!r}"
        )
    return raw


def validate_profile_model(hf_config: Any | None) -> None:
    """Fail closed when the explicit profile is used with another geometry."""

    if active_profile() is None:
        return
    if hf_config is None:
        raise ValueError(
            f"{PROFILE_ENV}={QWEN38_27B_PROFILE} requires a model configuration"
        )
    try:
        config = hf_config.get_text_config()
    except (AttributeError, TypeError):
        config = hf_config
    layers = getattr(config, "num_hidden_layers", None)
    intermediate = getattr(config, "intermediate_size", None)
    problems: list[str] = []
    if layers != 64:
        problems.append(f"num_hidden_layers={layers!r}")
    if intermediate != 17408:
        problems.append(f"intermediate_size={intermediate!r}")
    if problems:
        raise ValueError(
            f"{PROFILE_ENV}={QWEN38_27B_PROFILE} requires the validated "
            "Qwen3.8-27B Dense geometry; " + ", ".join(problems)
        )


def _matches_suffix(prefix: str, suffix: str) -> bool:
    name = prefix.lower()
    return name == suffix.removeprefix(".") or name.endswith(suffix)


def route_for_prefix(prefix: str) -> HybridRoute | None:
    if active_profile() is None:
        return None
    if any(_matches_suffix(prefix, suffix) for suffix in _ALL_ROWS_SUFFIXES):
        return HybridRoute.ALL_ROWS
    if any(_matches_suffix(prefix, suffix) for suffix in _PREFILL_ONLY_SUFFIXES):
        return HybridRoute.PREFILL_ONLY
    return None


def _logical_output_size(layer: torch.nn.Module, shard_id: Any) -> int:
    sizes = layer.exl3_output_partition_sizes
    if shard_id is None:
        return int(sizes[0])
    if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
        return int(sizes[{"q": 0, "k": 1, "v": 2}[shard_id]])
    if isinstance(shard_id, tuple):
        return sum(int(sizes[index]) for index in shard_id)
    if isinstance(shard_id, int):
        return int(sizes[shard_id])
    return int(sizes[layer.exl3_shard_ids.index(shard_id)])


def _load_runtime(device: torch.device) -> Any:
    try:
        installed = version("mxfp6-sm120")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{PROFILE_ENV} requires mxfp6-sm120=={_SUPPORTED_RUNTIME_VERSION}"
        ) from exc
    if installed != _SUPPORTED_RUNTIME_VERSION:
        raise RuntimeError(
            f"{PROFILE_ENV} requires mxfp6-sm120=={_SUPPORTED_RUNTIME_VERSION}; "
            f"found {installed}"
        )
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError(f"{PROFILE_ENV} currently requires an SM120 CUDA device")

    runtime = importlib.import_module("mxfp6")
    required = ("is_available", "load_library", "quantize_mxfp6")
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise RuntimeError(
            "mxfp6-sm120 is missing the Hybrid API: " + ", ".join(missing)
        )
    runtime.load_library()
    if not runtime.is_available():
        raise RuntimeError("mxfp6-sm120 is not available on the active device")
    anchor = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        abi = torch.ops.mxfp6.w6a8_config_abi(anchor)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("mxfp6-sm120 does not expose its W6A8 ABI") from exc
    if abi != _SUPPORTED_W6A8_ABI:
        raise RuntimeError(
            f"mxfp6-sm120 W6A8 ABI mismatch: expected {_SUPPORTED_W6A8_ABI!r}, "
            f"found {abi!r}"
        )
    return runtime


def _reconstruct_shard(
    layer: torch.nn.Module,
    extension: Any,
    shard_id: Any,
    logical_rows: int,
) -> torch.Tensor:
    if not hasattr(extension, "reconstruct_had_slice"):
        raise RuntimeError(
            "The EXL3/MXFP6 profile requires ExLlamaV3 reconstruct_had_slice"
        )
    trellis = layer.trellis.exl3_tensors[shard_id]
    suh = layer.suh.exl3_tensors[shard_id]
    svh = layer.svh.exl3_tensors[shard_id]
    k = int(trellis.shape[0]) * 16
    padded_rows = int(trellis.shape[1]) * 16
    if logical_rows > padded_rows:
        raise ValueError(
            f"EXL3 logical N={logical_rows} exceeds packed N={padded_rows}"
        )
    weight = torch.empty(
        (k, padded_rows),
        dtype=torch.float16,
        device=trellis.device,
    )
    extension.reconstruct_had_slice(
        weight,
        trellis,
        suh,
        svh,
        int(trellis.shape[2]) // 16,
        shard_id in layer.mcg.exl3_tensors,
        shard_id in layer.mul1.exl3_tensors,
        0,
    )
    return weight[:, :logical_rows]


def _quantize(runtime: Any, weight_k_n: torch.Tensor) -> HybridPackedWeight:
    k, rows = map(int, weight_k_n.shape)
    if rows <= 0 or rows % 8 != 0 or k <= 0 or k % 128 != 0:
        raise ValueError(
            "EXL3/MXFP6 requires N divisible by 8 and K divisible by 128; "
            f"got N={rows}, K={k}"
        )
    packed = runtime.quantize_mxfp6(weight_k_n.t().contiguous().to(torch.bfloat16))
    if int(packed.rows) != rows or int(packed.k) != k:
        raise RuntimeError(
            "mxfp6-sm120 returned an inconsistent packed shape: "
            f"expected ({rows}, {k}), got ({packed.rows}, {packed.k})"
        )
    return HybridPackedWeight(
        values=packed.values,
        scales=packed.scales,
        rows=rows,
        k=k,
    )


def prepare_layer(layer: torch.nn.Module, extension: Any) -> HybridState | None:
    """Build the rank-local MXFP6 copy for one selected EXL3 linear."""

    existing = getattr(layer, _STATE_ATTRIBUTE, None)
    if existing is not None:
        return existing
    prefix = str(getattr(layer, "prefix", ""))
    route = route_for_prefix(prefix)
    if route is None:
        return None
    if "lm_head" in prefix.lower() or type(layer).__name__ == "ParallelLMHead":
        raise ValueError("The EXL3/MXFP6 profile never converts lm_head")
    shard_ids = list(layer.exl3_shard_ids)
    if not shard_ids:
        raise ValueError(f"EXL3/MXFP6 selected {prefix!r} without any shards")

    first = layer.trellis.exl3_tensors[shard_ids[0]]
    runtime = _load_runtime(first.device)
    if prefix.lower().endswith("mlp.gate_up_proj"):
        if route is not HybridRoute.ALL_ROWS or len(shard_ids) != 2:
            raise ValueError(
                f"EXL3/MXFP6 gate/up expects two all-row shards; got {shard_ids!r}"
            )
        parts = [
            _reconstruct_shard(
                layer,
                extension,
                shard_id,
                _logical_output_size(layer, shard_id),
            )
            for shard_id in shard_ids
        ]
        if len({int(part.shape[0]) for part in parts}) != 1:
            raise ValueError("EXL3/MXFP6 gate/up shards have different K dimensions")
        merged = _quantize(runtime, torch.cat(parts, dim=1))
        state = HybridState(route=route, weights={}, merged_weight=merged)
    else:
        weights = {
            shard_id: _quantize(
                runtime,
                _reconstruct_shard(
                    layer,
                    extension,
                    shard_id,
                    _logical_output_size(layer, shard_id),
                ),
            )
            for shard_id in shard_ids
        }
        state = HybridState(route=route, weights=weights)

    setattr(layer, _STATE_ATTRIBUTE, state)
    if route is HybridRoute.ALL_ROWS:
        for name in ("trellis", "suh", "svh", "mcg", "mul1"):
            getattr(layer, name).exl3_tensors.clear()
    return state


def state_for_rows(layer: torch.nn.Module, rows: int) -> HybridState | None:
    state = getattr(layer, _STATE_ATTRIBUTE, None)
    if state is None or not state.active_for_rows(rows):
        return None
    return state


def iter_packed_weights(layer: torch.nn.Module) -> Iterable[HybridPackedWeight]:
    state = getattr(layer, _STATE_ATTRIBUTE, None)
    if state is None:
        return ()
    if state.merged_weight is not None:
        return (state.merged_weight,)
    return tuple(state.weights.values())


def apply_weight(x: torch.Tensor, weight: HybridPackedWeight) -> torch.Tensor:
    """Run the shared graph-safe vLLM Mach W6A8 custom operator."""

    from vllm_mach.mxfp6.dense import _ensure_custom_op_registered

    if int(x.shape[-1]) != weight.k:
        raise ValueError(
            f"EXL3/MXFP6 activation K={x.shape[-1]} does not match weight K={weight.k}"
        )
    _ensure_custom_op_registered()
    return torch.ops.vllm.mach_mxfp6_sm120_gemm(
        x.reshape(-1, weight.k).to(torch.bfloat16).contiguous(),
        weight.values,
        weight.scales,
        weight.rows,
        torch.bfloat16,
    )


def fused_silu_mxfp8(gate_up: torch.Tensor) -> Any:
    """Fuse SiLU/mul with the packed MXFP8 input for an MXFP6 down projection."""

    if gate_up.ndim != 2 or gate_up.dtype != torch.bfloat16:
        raise ValueError("fused SiLU/MXFP8 expects a two-dimensional BF16 tensor")
    rows, doubled_k = map(int, gate_up.shape)
    if doubled_k % 2:
        raise ValueError(f"gate/up width must be even; got {doubled_k}")
    k = doubled_k // 2
    if k % 32:
        raise ValueError(f"fused SiLU/MXFP8 requires K divisible by 32; got {k}")

    runtime = importlib.import_module("mxfp6")
    required = ("MXFP8Tensor", "silu_and_mul_mxfp8_packed_out")
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise RuntimeError(
            "mxfp6-sm120 is missing the fused MLP API: " + ", ".join(missing)
        )

    values = torch.empty((rows, k), dtype=torch.uint8, device=gate_up.device)
    scale_groups = ((k // 32 + 3) // 4) * 4
    scales = torch.empty(
        ((rows + 127) // 128 * 128) * scale_groups,
        dtype=torch.uint8,
        device=gate_up.device,
    )
    runtime.silu_and_mul_mxfp8_packed_out(values, scales, gate_up)
    return runtime.MXFP8Tensor(values, scales, rows, k)


def apply_mxfp8_weight(activation: Any, weight: HybridPackedWeight) -> torch.Tensor:
    """Apply one Hybrid weight to an already packed MXFP8 activation."""

    runtime = importlib.import_module("mxfp6")
    required = ("PackedMXFP6Tensor", "gemm_w6a8")
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise RuntimeError(
            "mxfp6-sm120 is missing the packed W6A8 API: " + ", ".join(missing)
        )
    if int(activation.k) != weight.k:
        raise ValueError(
            f"MXFP8 activation K={activation.k} does not match weight K={weight.k}"
        )
    packed_weight = runtime.PackedMXFP6Tensor(
        weight.values,
        weight.scales,
        weight.rows,
        weight.k,
    )
    return runtime.gemm_w6a8(
        activation,
        packed_weight,
        out_dtype=torch.bfloat16,
    )


__all__ = [
    "PROFILE_ENV",
    "QWEN38_27B_PROFILE",
    "HybridPackedWeight",
    "HybridRoute",
    "HybridState",
    "active_profile",
    "apply_mxfp8_weight",
    "apply_weight",
    "fused_silu_mxfp8",
    "iter_packed_weights",
    "prepare_layer",
    "route_for_prefix",
    "state_for_rows",
    "validate_profile_model",
]
