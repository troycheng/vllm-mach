# SPDX-License-Identifier: Apache-2.0
"""Portable, hash-bound assets for the selected TP2 rank64 gate/up path."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "vllm-mach-rank64/v1"
MASK = tuple(range(27)) + (34, 35, 36, 38, 42) + tuple(range(48, 64))
OFFICIAL_CONFIG = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
OFFICIAL_INDEX = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
SCALE_RECEIPT = "8244f9c83ae2d1c787146bb3dfc6a7e4c60fcb78bc59d13f1f957c766d80a795"
SCALE_CONTENT = "c4a3bb7f63a966d7498b28e40448f0f139bc929da8db90f24dcffda31d3a8645"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value) -> str:
    import torch
    return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def static_scale_content(data: dict) -> str:
    """Bind all norm identities and selected scales, excluding report paths."""
    value = {"config": data["config"], "layers": {
        key: {field: row[field] for field in (
            "raw_payload_sha256", "selected_power_of_two_global_scale")}
        for key, row in data["layers"].items()}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_static_scales(path: Path) -> dict:
    data = json.loads(path.read_text())
    if static_scale_content(data) != SCALE_CONTENT:
        raise ValueError("static scale numerical content or RMSNorm identity differs from the selected model")
    return data


def checked_path(root: Path, name: str, digest: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("rank64 asset must be a relative path below its bundle")
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("rank64 asset escapes its bundle")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("rank64 asset has an invalid SHA256")
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"rank64 asset is missing or SHA256 differs: {path}")
    return path


def load_manifest(root: Path) -> dict:
    if not root.is_absolute():
        raise ValueError("VLLM_MACH_RANK64_BUNDLE must be an absolute directory")
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"rank64 bundle needs {path}; run tools/import_rank64_bundle.py")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("rank64 manifest must be a JSON object")
    if data.get("schema") != SCHEMA or data.get("mask") != list(MASK):
        raise ValueError("rank64 bundle must use the selected 48-layer mask and v1 schema")
    if data.get("official_config_sha256") != OFFICIAL_CONFIG or data.get("official_index_sha256") != OFFICIAL_INDEX:
        raise ValueError("rank64 bundle must originate from the selected Qwen3.8-27B checkpoint")
    if data.get("geometry") != {"rows": 32, "hidden": 5120, "gate_up": 17408, "rank": 64, "tp": 2}:
        raise ValueError("rank64 bundle geometry mismatch")
    entries = data.get("entries", [])
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("rank64 entries must be a list of objects")
    keys = [(item.get("layer"), item.get("rank")) for item in entries]
    expected = {(layer, rank) for layer in MASK for rank in (0, 1)}
    if len(keys) != 96 or set(keys) != expected or any(type(v) is not int for key in keys for v in key):
        raise ValueError("rank64 bundle must cover exactly 48 layers x TP2 without duplicates")
    scales_path = checked_path(root, "static_scales.json", data.get("static_scales_sha256", SCALE_RECEIPT))
    scales = validate_static_scales(scales_path)["layers"]
    for item in entries:
        if item.get("activation_global_scale") not in (16.0, 32.0):
            raise ValueError("rank64 activation scale must be the selected analytic 16 or 32")
        for key in ("official_bf16_gu_sha256", "norm_bf16_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(item.get(key, ""))) is None:
                raise ValueError(f"rank64 bundle needs {key}")
        scale = scales[str(item["layer"])]
        if item["activation_global_scale"] != scale["selected_power_of_two_global_scale"] or item["norm_bf16_sha256"] != scale["raw_payload_sha256"]:
            raise ValueError("rank64 activation scale/RMSNorm identity differs from its static scale receipt")
    return data


def load_entry(root: Path, entry: dict) -> dict:
    """Validate CPU payload bytes before copying any tensors to a GPU."""
    import torch
    tensors = {}
    shapes = {"packed": (17408, 2560), "scales": (17408, 320),
              "global_scale": (1,), "aware64_a": (5120, 64), "aware64_b": (64, 17408)}
    dtypes = {"packed": torch.uint8, "scales": torch.float8_e4m3fn,
              "global_scale": torch.float32, "aware64_a": torch.bfloat16,
              "aware64_b": torch.bfloat16}
    for group, names in (("weight", ("packed", "scales", "global_scale")),
                         ("residual", ("aware64_a", "aware64_b"))):
        meta = entry[group]
        path = checked_path(root, meta["file"], meta["file_sha256"])
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload.get("layer") != entry["layer"] or payload.get("rank") != entry["rank"]:
            raise ValueError(f"rank64 {group} payload layer/rank mismatch")
        if group == "weight" and payload.get("official_bf16_gu_sha256") != entry["official_bf16_gu_sha256"]:
            raise ValueError("rank64 weight checkpoint identity mismatch")
        for name in names:
            value = payload[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name] or value.dtype != dtypes[name] or not value.is_contiguous():
                raise ValueError(f"rank64 {name} shape/dtype/layout mismatch")
            if tensor_sha256(value) != meta["tensor_sha256"][name]:
                raise ValueError(f"rank64 {name} SHA256 mismatch")
            if name != "packed" and not torch.isfinite(value.float()).all():
                raise ValueError(f"rank64 {name} contains non-finite values")
            tensors[name] = value
    if float(tensors["global_scale"].item()) <= 0:
        raise ValueError("rank64 weight global scale must be positive")
    return tensors
