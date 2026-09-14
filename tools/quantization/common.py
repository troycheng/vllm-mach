# SPDX-License-Identifier: Apache-2.0
"""Shared file contracts for offline model-asset generation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


assets = module("mach_generation_assets", ROOT / "src/vllm_mach/exl3/rank64_assets.py")
sha = assets.sha256


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def checked_model(path):
    path = Path(path).resolve()
    if sha(path / "config.json") != assets.OFFICIAL_CONFIG or sha(path / "model.safetensors.index.json") != assets.OFFICIAL_INDEX:
        raise ValueError("Expected original Qwen3.8-27B config/index at ModelScope revision e823e888ae179eb3be02c1a48899c4f828371376")
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(index.values()):
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file():
            raise ValueError(f"Missing or unsafe source shard: {name}")
    return index


def train_rows(capture, samples, layer, offsets):
    import torch
    ids = sorted(s["id"] for s in samples if s["split"] == "train")
    # QA order follows the historical task/hash order, not lexical sample order.
    if offsets == list(range(1, 9)):
        ids = []
        for task in sorted({s["task"] for s in samples}):
            group = sorted((s["id"] for s in samples if s["task"] == task),
                           key=lambda x: hashlib.sha256(("hessian-pilot-v1:" + x).encode()).digest())
            ids.extend(group[:2])
        if set(ids) != {s["id"] for s in samples if s["split"] == "train"}:
            raise ValueError("QA training split changed")
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError("Expected 16 distinct training samples")
    calls = capture["layers"][f"language_model.model.layers.{layer}.mlp.gate_up_proj"]["calls"]
    if [c["decode_offset"] for c in calls] != offsets:
        raise ValueError("Calibration decode offsets changed")
    rows = []
    for call in calls:
        mapping = call["sample_ids"]
        if len(mapping) != 32 or len(set(mapping)) != 32 or set(mapping) != {s["id"] for s in samples}:
            raise ValueError("Calibration sample mapping changed")
        x = call["x"]
        if x.dtype != torch.bfloat16 or tuple(x.shape) != (32, 5120) or not torch.isfinite(x).all():
            raise ValueError("Expected finite BF16 [32,5120] calibration input")
        positions = {sample: i for i, sample in enumerate(mapping)}
        rows.append(x[[positions[sample] for sample in ids]])
    return torch.cat(rows).contiguous()
