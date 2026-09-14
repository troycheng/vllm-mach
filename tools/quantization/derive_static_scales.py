#!/usr/bin/env python3
"""Read only Qwen3.8 RMSNorm BF16 payloads and derive static NVFP4 A scales.

This extractor intentionally uses only the Python standard library.  It reads
the safetensors header plus the 10,240 selected bytes for each of 64
post-attention RMSNorm weights; it never reads or hashes whole model shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_LAYERS = 64
MAX_HEADER_BYTES = 100_000_000
NVFP4_NUMERATOR = 448.0 * 6.0
FP32_UNIT_ROUNDOFF = 2.0**-24
BF16_UNIT_ROUNDOFF = 2.0**-8
E4M3_MIN_SUBNORMAL = 2.0**-9
E4M3_SCALE_ZERO_MIDPOINT = E4M3_MIN_SUBNORMAL / 2.0
KEY_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\."
    r"post_attention_layernorm\.weight$"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_bytes(data: bytes, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON must be an object: {source}")
    return value


def _read_small_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return _json_bytes(data, path), _sha256_bytes(data)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"truncated safetensors length prefix: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length > MAX_HEADER_BYTES:
            raise ValueError(
                f"safetensors header exceeds {MAX_HEADER_BYTES} bytes: {path}"
            )
        data_start = 8 + header_length
        if data_start > file_size:
            raise ValueError(
                f"safetensors header exceeds file: header_end={data_start}, "
                f"file_size={file_size}, path={path}"
            )
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"short safetensors header read: {path}")
    header = _json_bytes(header_bytes, path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"__metadata__ must be an object: {path}")

    tensors: dict[str, dict[str, Any]] = {}
    data_size = file_size - data_start
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise ValueError(f"tensor descriptor must be an object: {path}:{name}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"invalid dtype: {path}:{name}")
        if not isinstance(shape, list) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape
        ):
            raise ValueError(f"invalid shape: {path}:{name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets)
        ):
            raise ValueError(f"invalid data_offsets: {path}:{name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(
                f"out-of-bounds data_offsets={offsets}, data_size={data_size}: "
                f"{path}:{name}"
            )
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }
    return {
        "path": path,
        "file_size": file_size,
        "header_length": header_length,
        "data_start": data_start,
        "header_sha256": _sha256_bytes(prefix + header_bytes),
        "metadata_sha256": _canonical_hash(metadata),
        "tensors": tensors,
    }


def _safe_shard_path(model_dir: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"invalid shard filename: {filename!r}")
    relative = Path(filename)
    if relative.is_absolute() or relative.name != filename:
        raise ValueError(f"shard filename must be a basename: {filename!r}")
    path = model_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _discover_selected(
    model_dir: Path,
) -> tuple[dict[int, tuple[str, Path]], Path | None, str | None]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"ambiguous safetensors indexes: {indexes}")
    selected: dict[int, tuple[str, Path]] = {}
    if indexes:
        index, index_sha256 = _read_small_json(indexes[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"index weight_map must be an object: {indexes[0]}")
        for name, filename in weight_map.items():
            match = KEY_RE.match(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in selected:
                raise ValueError(f"duplicate post-attention norm layer {layer}")
            selected[layer] = (name, _safe_shard_path(model_dir, filename))
        return selected, indexes[0], index_sha256

    # Index-free fallback scans only shard headers.
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {model_dir}")
    for shard in shards:
        header = _read_safetensors_header(shard)
        for name in header["tensors"]:
            match = KEY_RE.match(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in selected:
                raise ValueError(f"duplicate post-attention norm layer {layer}")
            selected[layer] = (name, shard)
    return selected, None, None


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16_summary(raw: bytes, source: str) -> dict[str, float]:
    if len(raw) != EXPECTED_HIDDEN_SIZE * 2:
        raise ValueError(f"wrong BF16 payload length for {source}: {len(raw)}")
    raw_min = math.inf
    raw_max = -math.inf
    gamma = 0.0
    for (bits,) in struct.iter_unpack("<H", raw):
        value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        if not math.isfinite(value):
            raise ValueError(f"non-finite BF16 norm weight in {source}: 0x{bits:04x}")
        # BF16 is exactly representable in FP32.  This explicit pack/unpack
        # rounds the +1 operation to FP32, matching weight.float() + 1.0.
        one_plus = _f32(value + 1.0)
        if not math.isfinite(one_plus):
            raise ValueError(f"non-finite FP32 (1+weight) in {source}")
        raw_min = min(raw_min, value)
        raw_max = max(raw_max, value)
        gamma = max(gamma, abs(one_plus))
    if gamma <= 0.0:
        raise ValueError(f"degenerate all-zero (1+weight) in {source}")
    return {
        "raw_weight_min": raw_min,
        "raw_weight_max": raw_max,
        "max_abs_one_plus_weight": gamma,
    }


def _floor_power_of_two(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid scale limit: {value}")
    return math.ldexp(1.0, math.floor(math.log2(value)))


def derive(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    config, config_sha256 = _read_small_json(config_path)
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    hidden_size = int(text["hidden_size"])
    num_layers = int(text["num_hidden_layers"])
    eps = float(text["rms_norm_eps"])
    if hidden_size != EXPECTED_HIDDEN_SIZE or num_layers != EXPECTED_LAYERS:
        raise ValueError(
            f"requires Qwen3.8 geometry hidden={EXPECTED_HIDDEN_SIZE}, "
            f"layers={EXPECTED_LAYERS}; got hidden={hidden_size}, layers={num_layers}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"invalid rms_norm_eps: {eps}")

    selected, index_path, index_sha256 = _discover_selected(model_dir)
    expected = set(range(EXPECTED_LAYERS))
    if set(selected) != expected:
        raise ValueError(
            f"need exactly norm layers 0..{EXPECTED_LAYERS - 1}; "
            f"got {sorted(selected)}"
        )

    headers = {
        path: _read_safetensors_header(path)
        for path in sorted({entry[1] for entry in selected.values()})
    }
    fp32_gamma = (hidden_size + 4) * FP32_UNIT_ROUNDOFF / (
        1.0 - (hidden_size + 4) * FP32_UNIT_ROUNDOFF
    )
    theory_margin = (1.0 / math.sqrt(1.0 - fp32_gamma)) * (
        1.0 + BF16_UNIT_ROUNDOFF
    )
    arithmetic_margin = max(theory_margin, 1.01)

    layers: dict[str, dict[str, Any]] = {}
    per_shard_count = {path: 0 for path in headers}
    open_handles = {path: path.open("rb") for path in headers}
    try:
        for layer in range(EXPECTED_LAYERS):
            name, shard = selected[layer]
            header = headers[shard]
            descriptor = header["tensors"].get(name)
            if descriptor is None:
                raise ValueError(f"index/header mismatch: {shard}:{name}")
            if descriptor["dtype"] != "BF16" or descriptor["shape"] != [hidden_size]:
                raise ValueError(
                    f"requires BF16 [{hidden_size}]: {shard}:{name}, got {descriptor}"
                )
            start, end = descriptor["data_offsets"]
            if end - start != hidden_size * 2:
                raise ValueError(f"wrong selected byte length: {shard}:{name}")
            handle = open_handles[shard]
            handle.seek(header["data_start"] + start)
            raw = handle.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"short selected payload read: {shard}:{name}")
            per_shard_count[shard] += 1
            stats = _bf16_summary(raw, f"{shard.name}:{name}")
            gamma = stats["max_abs_one_plus_weight"]
            exact_bound = math.sqrt(hidden_size) * gamma
            guarded_bound = exact_bound * arithmetic_margin
            scale_limit = NVFP4_NUMERATOR / guarded_bound
            fixed_global_scale = _floor_power_of_two(scale_limit)
            layers[str(layer)] = {
                "name": name,
                "source_shard": shard.name,
                "dtype": "BF16",
                "shape": {"rank": 1, "dim0": hidden_size},
                "raw_payload_sha256": _sha256_bytes(raw),
                **stats,
                "exact_real_rmsnorm_output_bound": exact_bound,
                "fp32_bf16_guarded_bound": guarded_bound,
                "safe_global_scale_limit": scale_limit,
                "selected_power_of_two_global_scale": fixed_global_scale,
                "worst_case_e4m3_block_scale": (
                    fixed_global_scale * guarded_bound / 6.0
                ),
                "block_amax_below_this_can_round_scale_to_zero": (
                    6.0 * E4M3_SCALE_ZERO_MIDPOINT / fixed_global_scale
                ),
            }
    finally:
        for handle in open_handles.values():
            handle.close()

    script_path = Path(__file__).resolve()
    return {
        "schema": "qwen38-gemma-rmsnorm-stdlib-static-nvfp4-scale/v1",
        "source": {
            "script": str(script_path),
            "script_sha256": _sha256_file(script_path),
            "config": str(config_path),
            "config_sha256": config_sha256,
            "index": str(index_path) if index_path is not None else None,
            "index_sha256": index_sha256,
            "used_shard_headers": {
                path.name: {
                    "file_size": header["file_size"],
                    "header_length": header["header_length"],
                    "header_sha256": header["header_sha256"],
                    "metadata_sha256": header["metadata_sha256"],
                    "selected_tensor_count": per_shard_count[path],
                }
                for path, header in headers.items()
            },
        },
        "config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "rms_norm_eps": eps,
        },
        "formula": {
            "norm": "BF16((x / sqrt(mean(x^2)+eps)) * FP32(1+raw_weight))",
            "real_bound": "sqrt(hidden_size) * max(abs(FP32(1+raw_weight)))",
            "fp32_gamma_d_plus_4": fp32_gamma,
            "theory_rounding_margin": theory_margin,
            "non_data_derived_arithmetic_margin": arithmetic_margin,
            "implementation_guard_is_formal_intrinsic_proof": False,
            "selection": "floor_power_of_two(2688 / guarded_bound)",
            "uses_samples_captures_or_gpu": False,
            "e4m3_max": 448.0,
            "e2m1_max": 6.0,
            "e4m3_min_subnormal": E4M3_MIN_SUBNORMAL,
        },
        "payload_policy": {
            "whole_shards_read_or_hashed": False,
            "selected_payload_bytes": EXPECTED_LAYERS * EXPECTED_HIDDEN_SIZE * 2,
            "raw_values_in_receipt": False,
        },
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = derive(args.model_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
