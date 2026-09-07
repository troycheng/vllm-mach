# SPDX-License-Identifier: Apache-2.0

"""Load original Qwen3.8 MXFP6 checkpoint bytes for EXL3 hybrid layers.

The loader only translates checkpoint tensor slicing into the packed-weight
container used by :mod:`vllm_mach.exl3.mxfp6_hybrid`.  It never dequantizes or
requantizes weights.  Scale packing is delegated to the installed MXFP6
runtime so its public ABI remains the single source of truth.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .mxfp6_hybrid import HybridPackedWeight

_INDEX_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")

_GEOMETRY = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "attn_output_gate": True,
}
_WEIGHT_QUANT = {
    "dtype": "fp6_e3m2",
    "group_size": 32,
    "is_dynamic": False,
    "qscheme": "per_group",
    "scale_format": "e8m0",
    "symmetric": True,
}
_INPUT_QUANT = {
    "dtype": "fp8_e4m3",
    "group_size": 32,
    "is_dynamic": True,
    "qscheme": "per_group",
    "scale_format": "e8m0",
    "symmetric": True,
}
_SOURCE_SHAPES = {
    "linear_attn.in_proj_qkv": (10240, 5120),
    "linear_attn.in_proj_z": (6144, 5120),
    "self_attn.q_proj": (12288, 5120),
    "self_attn.k_proj": (1024, 5120),
    "self_attn.v_proj": (1024, 5120),
    "mlp.gate_proj": (17408, 5120),
    "mlp.up_proj": (17408, 5120),
    "mlp.down_proj": (5120, 17408),
    "linear_attn.out_proj": (5120, 6144),
    "self_attn.o_proj": (5120, 6144),
}


@dataclass(frozen=True)
class _PartSpec:
    key: str
    source_rows: int
    source_k: int
    row_start: int
    rows: int
    k_start: int
    k: int


@dataclass(frozen=True)
class _LayerPlan:
    shard_ids: tuple[Any, ...]
    parts: tuple[_PartSpec, ...]
    merged: bool


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Rank-local checkpoint weights and their EXL3 logical shard order."""

    shard_ids: tuple[Any, ...]
    weights: dict[Any, HybridPackedWeight]
    merged_weight: HybridPackedWeight | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _get(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _text_config(config: Any) -> Any:
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        return getter()
    nested = _get(config, "text_config")
    return config if nested is None else nested


def _expected_layer_types() -> list[str]:
    return [
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(_GEOMETRY["num_hidden_layers"])
    ]


def _validate_geometry(config: Any, *, source: str) -> None:
    text = _text_config(config)
    problems = [
        f"{name}={_get(text, name)!r}"
        for name, expected in _GEOMETRY.items()
        if _get(text, name) != expected
    ]
    if _get(text, "layer_types") != _expected_layer_types():
        problems.append("layer_types does not match the Qwen3.8-27B profile")
    if problems:
        raise ValueError(
            f"{source} is not the validated Qwen3.8-27B geometry: "
            + ", ".join(problems)
        )


def _validate_quantization(config: Mapping[str, Any]) -> None:
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise ValueError("checkpoint config has no quantization_config object")
    global_quant = quant.get("global_quant_config")
    if not isinstance(global_quant, Mapping):
        raise ValueError("checkpoint config has no global_quant_config object")
    export = quant.get("export")
    problems: list[str] = []
    if quant.get("quant_method") != "quark":
        problems.append(f"quant_method={quant.get('quant_method')!r}")
    for override_name in ("layer_quant_config", "layer_type_quant_config"):
        if quant.get(override_name, {}) != {}:
            problems.append(f"{override_name} must be empty")
    if not isinstance(export, Mapping) or export.get("pack_method") != "reorder":
        pack_method = None if not isinstance(export, Mapping) else export.get(
            "pack_method"
        )
        problems.append(f"export.pack_method={pack_method!r}")
    for name, expected in (
        ("weight", _WEIGHT_QUANT),
        ("input_tensors", _INPUT_QUANT),
    ):
        actual = global_quant.get(name)
        if not isinstance(actual, Mapping):
            problems.append(f"global_quant_config.{name} is missing")
            continue
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                problems.append(
                    f"global_quant_config.{name}.{field}={actual.get(field)!r}"
                )
    if problems:
        raise ValueError(
            "checkpoint quantization metadata is incompatible with native "
            "MXFP6 W6A8: " + ", ".join(problems)
        )


def _safe_shard_path(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"checkpoint index contains invalid filename {filename!r}")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != filename
        or relative.suffix != ".safetensors"
    ):
        raise ValueError(f"unsafe checkpoint shard filename {filename!r}")
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"checkpoint shard does not exist: {filename!r}") from exc
    if resolved.parent != root.resolve() or not resolved.is_file():
        raise ValueError(f"unsafe checkpoint shard filename {filename!r}")
    return resolved


def _part(key: str, row_start: int, rows: int, k_start: int, k: int) -> _PartSpec:
    suffix = key.split(".layers.", 1)[1].split(".", 1)[1]
    source_rows, source_k = _SOURCE_SHAPES[suffix]
    return _PartSpec(key, source_rows, source_k, row_start, rows, k_start, k)


def _layer_plan(prefix: str, rank: int) -> _LayerPlan:
    match = _LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(f"unsupported checkpoint projection: {prefix!r}")
    layer_index_text, suffix = match.groups()
    layer_index = int(layer_index_text)
    if not 0 <= layer_index < _GEOMETRY["num_hidden_layers"]:
        raise ValueError(f"checkpoint layer index is out of range: {layer_index}")
    if suffix.startswith("linear_attn.") and layer_index % 4 == 3:
        raise ValueError(f"layer {layer_index} is not a linear-attention layer")
    if suffix.startswith("self_attn.") and layer_index % 4 != 3:
        raise ValueError(f"layer {layer_index} is not a full-attention layer")

    base = f"model.language_model.layers.{layer_index}."
    if suffix == "linear_attn.in_proj_qkvz":
        qkv = base + "linear_attn.in_proj_qkv"
        z = base + "linear_attn.in_proj_z"
        return _LayerPlan(
            (0, 1, 2, 3),
            (
                _part(qkv, rank * 1024, 1024, 0, 5120),
                _part(qkv, 2048 + rank * 1024, 1024, 0, 5120),
                _part(qkv, 4096 + rank * 3072, 3072, 0, 5120),
                _part(z, rank * 3072, 3072, 0, 5120),
            ),
            True,
        )
    if suffix == "self_attn.qkv_proj":
        return _LayerPlan(
            ("q", "k", "v"),
            (
                _part(base + "self_attn.q_proj", rank * 6144, 6144, 0, 5120),
                _part(base + "self_attn.k_proj", rank * 512, 512, 0, 5120),
                _part(base + "self_attn.v_proj", rank * 512, 512, 0, 5120),
            ),
            True,
        )
    if suffix == "mlp.gate_up_proj":
        return _LayerPlan(
            (0, 1),
            (
                _part(base + "mlp.gate_proj", rank * 8704, 8704, 0, 5120),
                _part(base + "mlp.up_proj", rank * 8704, 8704, 0, 5120),
            ),
            True,
        )
    if suffix == "mlp.down_proj":
        return _LayerPlan(
            (None,),
            (_part(base + suffix, 0, 5120, rank * 8704, 8704),),
            False,
        )
    if suffix in ("linear_attn.out_proj", "self_attn.o_proj"):
        return _LayerPlan(
            (None,),
            (_part(base + suffix, 0, 5120, rank * 3072, 3072),),
            False,
        )
    raise ValueError(f"unsupported checkpoint projection: {prefix!r}")


class Mxfp6Checkpoint:
    """Validated, root-bound reader for the Qwen3.8-27B MXFP6 checkpoint."""

    def __init__(self, root: str | Path, expected_config: Any | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"MXFP6 checkpoint root is not a directory: {self.root}")

        config = _read_json_object(self.root / _CONFIG_NAME)
        _validate_geometry(config, source="MXFP6 checkpoint")
        _validate_quantization(config)
        if expected_config is not None:
            _validate_geometry(expected_config, source="served model")

        index = _read_json_object(self.root / _INDEX_NAME)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{_INDEX_NAME} has no non-empty weight_map object")
        self._weight_map: dict[str, str] = {}
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError("checkpoint index contains an invalid tensor name")
            _safe_shard_path(self.root, filename)
            self._weight_map[tensor_name] = filename

    def _read_part(self, spec: _PartSpec) -> tuple[torch.Tensor, torch.Tensor]:
        if spec.k_start % 32 or spec.k % 32 or spec.rows % 8:
            raise ValueError(
                "MXFP6 slices require K coordinates divisible by 32 and N "
                f"divisible by 8; got {spec!r}"
            )
        if (
            spec.row_start < 0
            or spec.k_start < 0
            or spec.rows <= 0
            or spec.k <= 0
            or spec.row_start + spec.rows > spec.source_rows
            or spec.k_start + spec.k > spec.source_k
        ):
            raise ValueError(f"MXFP6 slice is outside its source tensor: {spec!r}")

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "loading an MXFP6 checkpoint requires safetensors"
            ) from exc

        outputs: list[torch.Tensor] = []
        for suffix, factor, divisor in (
            (".weight", 3, 4),
            (".weight_scale", 1, 32),
        ):
            name = spec.key + suffix
            try:
                filename = self._weight_map[name]
            except KeyError as exc:
                raise ValueError(
                    f"checkpoint index is missing tensor {name!r}"
                ) from exc
            path = _safe_shard_path(self.root, filename)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if name not in handle.keys():
                    raise ValueError(
                        f"checkpoint shard {filename!r} is missing tensor {name!r}"
                    )
                source = handle.get_slice(name)
                expected_shape = (
                    spec.source_rows,
                    spec.source_k * factor // divisor,
                )
                shape = tuple(source.get_shape())
                if shape != expected_shape:
                    raise ValueError(
                        f"checkpoint tensor {name!r} has shape {shape}; "
                        f"expected {expected_shape}"
                    )
                column_start = spec.k_start * factor // divisor
                columns = spec.k * factor // divisor
                tensor = source[
                    spec.row_start : spec.row_start + spec.rows,
                    column_start : column_start + columns,
                ].contiguous()
            expected_slice_shape = (spec.rows, columns)
            if (
                tensor.dtype is not torch.uint8
                or tuple(tensor.shape) != expected_slice_shape
            ):
                raise ValueError(
                    f"checkpoint tensor {name!r} must yield uint8 "
                    f"{expected_slice_shape}; got {tensor.dtype} {tuple(tensor.shape)}"
                )
            outputs.append(tensor)
        return outputs[0], outputs[1]

    @staticmethod
    def _pack_weight(
        parts: list[tuple[torch.Tensor, torch.Tensor, int, int]],
        *,
        device: torch.device,
        pack_scales: Callable[[torch.Tensor], torch.Tensor],
    ) -> HybridPackedWeight:
        if not parts:
            raise ValueError("cannot build an MXFP6 weight without checkpoint parts")
        ks = {k for _, _, _, k in parts}
        if len(ks) != 1:
            raise ValueError(
                "merged MXFP6 checkpoint parts have different K dimensions"
            )
        k = next(iter(ks))
        rows = sum(rows for _, _, rows, _ in parts)
        values = torch.cat([part[0] for part in parts], dim=0)
        logical_scales = torch.cat([part[1] for part in parts], dim=0)
        values = values.to(device=device).contiguous().view(-1)
        logical_scales = logical_scales.to(device=device).contiguous()
        scales = pack_scales(logical_scales)
        if not isinstance(scales, torch.Tensor):
            raise TypeError("pack_scales must return a torch.Tensor")
        if scales.dtype is not torch.uint8 or scales.device != device:
            raise ValueError(
                "pack_scales must return uint8 scales on the requested device; "
                f"got {scales.dtype} on {scales.device}"
            )
        return HybridPackedWeight(
            values=values,
            scales=scales.contiguous(),
            rows=rows,
            k=k,
        )

    def load_layer(
        self,
        prefix: str,
        *,
        rank: int,
        tp_size: int,
        device: torch.device | str,
        pack_scales: Callable[[torch.Tensor], torch.Tensor],
    ) -> CheckpointLoadResult:
        """Load one rank-local projection without changing its packed values."""

        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(f"tensor-parallel rank must be an integer; got {rank!r}")
        if isinstance(tp_size, bool) or tp_size != 2:
            raise ValueError(
                f"Qwen3.8 MXFP6 checkpoint loading requires TP=2; got {tp_size!r}"
            )
        if rank not in (0, 1):
            raise ValueError(
                f"Qwen3.8 MXFP6 checkpoint rank must be 0 or 1; got {rank!r}"
            )
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("checkpoint projection prefix must be a non-empty string")
        if not callable(pack_scales):
            raise TypeError("pack_scales must be callable")

        target_device = torch.device(device)
        plan = _layer_plan(prefix, rank)
        parts = []
        for spec in plan.parts:
            values, scales = self._read_part(spec)
            parts.append((values, scales, spec.rows, spec.k))

        if plan.merged:
            merged_weight = self._pack_weight(
                parts,
                device=target_device,
                pack_scales=pack_scales,
            )
            return CheckpointLoadResult(plan.shard_ids, {}, merged_weight)

        if len(parts) != 1 or len(plan.shard_ids) != 1:
            raise AssertionError("non-merged checkpoint plan must contain one shard")
        weight = self._pack_weight(
            parts,
            device=target_device,
            pack_scales=pack_scales,
        )
        return CheckpointLoadResult(plan.shard_ids, {plan.shard_ids[0]: weight})


__all__ = ["CheckpointLoadResult", "Mxfp6Checkpoint"]
