#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package existing selected GU weights, aware64 coefficients and static scales.

This imports already generated assets; it does not quantize or fit a model.
No checkpoint tensors are uploaded. The destination must not already exist.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

# Asset packaging needs CPU PyTorch, not an installed vLLM/CUDA extension.
_source = Path(__file__).resolve().parents[1] / "src/vllm_mach/exl3/rank64_assets.py"
_spec = importlib.util.spec_from_file_location("mach_rank64_assets", _source)
_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_assets)
MASK, OFFICIAL_CONFIG, OFFICIAL_INDEX = _assets.MASK, _assets.OFFICIAL_CONFIG, _assets.OFFICIAL_INDEX
SCALE_RECEIPT, SCHEMA = _assets.SCALE_RECEIPT, _assets.SCHEMA
checked_path, load_entry, load_manifest, sha256 = _assets.checked_path, _assets.load_entry, _assets.load_manifest, _assets.sha256


def import_bundle(weights_path: Path, residual_path: Path, scales_path: Path, output: Path) -> dict:
    import torch
    weights = json.loads(weights_path.read_text())
    residual = json.loads(residual_path.read_text())
    scales = _assets.validate_static_scales(scales_path)
    for name, manifest in (("weight", weights), ("residual", residual)):
        if manifest.get("schema") != 1 or manifest.get("mask") != list(MASK):
            raise ValueError(f"{name} manifest must contain the selected 48-layer mask")
        keys = [(e["layer"], e["rank"]) for e in manifest["entries"]]
        if len(keys) != 96 or set(keys) != {(layer, rank) for layer in MASK for rank in (0, 1)}:
            raise ValueError(f"{name} manifest must cover exactly 48 layers x TP2")
    if weights.get("official_config_sha256") != OFFICIAL_CONFIG or weights.get("official_index_sha256") != OFFICIAL_INDEX:
        raise ValueError("weight manifest checkpoint identity mismatch")
    by_key = {(e["layer"], e["rank"]): e for e in residual["entries"]}
    # Check the source linkage before creating the destination.
    for weight in weights["entries"]:
        correction = by_key[weight["layer"], weight["rank"]]
        hashes = {name: weight[f"{name}_sha256"] for name in ("packed", "scales", "global_scale")}
        if correction["base_file_sha256"] != weight["file_sha256"] or correction["base_tensor_sha256"] != hashes or correction["official_bf16_gu_sha256"] != weight["official_bf16_gu_sha256"]:
            raise ValueError("rank64 correction does not belong to its NVFP4 weight")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "weights").mkdir()
    (output / "residuals").mkdir()
    shutil.copyfile(scales_path, output / "static_scales.json")
    result = {
        "schema": SCHEMA, "mask": list(MASK),
        "geometry": {"rows": 32, "hidden": 5120, "gate_up": 17408, "rank": 64, "tp": 2},
        "official_config_sha256": OFFICIAL_CONFIG, "official_index_sha256": OFFICIAL_INDEX,
        "static_scales_sha256": sha256(scales_path),
        "source_manifests_sha256": {"weight": sha256(weights_path), "residual": sha256(residual_path), "scales": sha256(scales_path)},
        "entries": [],
    }
    for weight in weights["entries"]:
        layer, rank = weight["layer"], weight["rank"]
        correction = by_key[layer, rank]
        base_file = checked_path(weights_path.parent, weight["file"], weight["file_sha256"])
        residual_file = checked_path(residual_path.parent, correction["file"], correction["file_sha256"])
        weight_name = f"weights/layer{layer:02d}_rank{rank}.pt"
        residual_name = f"residuals/layer{layer:02d}_rank{rank}.pt"
        shutil.copyfile(base_file, output / weight_name)
        payload = torch.load(residual_file, weights_only=True, map_location="cpu")
        if payload.get("layer") != layer or payload.get("rank") != rank:
            raise ValueError("source residual payload layer/rank mismatch")
        # Omit unselected cols32 coefficients from the portable payload.
        torch.save({key: payload[key] for key in ("layer", "rank", "aware64_a", "aware64_b")}, output / residual_name)
        scale = scales["layers"][str(layer)]
        entry = {
            "layer": layer, "rank": rank,
            "official_bf16_gu_sha256": weight["official_bf16_gu_sha256"],
            "norm_bf16_sha256": scale["raw_payload_sha256"],
            "activation_global_scale": scale["selected_power_of_two_global_scale"],
            "weight": {"file": weight_name, "file_sha256": weight["file_sha256"],
                       "tensor_sha256": {name: weight[f"{name}_sha256"] for name in ("packed", "scales", "global_scale")}},
            "residual": {"file": residual_name, "file_sha256": sha256(output / residual_name),
                         "source_file_sha256": correction["file_sha256"],
                         "tensor_sha256": {name: correction["tensor_sha256"][name] for name in ("aware64_a", "aware64_b")}},
        }
        load_entry(output, entry)
        result["entries"].append(entry)
        print(f"validated layer {layer}, rank {rank}", flush=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    load_manifest(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="mask48_manifest.json with its payloads directory")
    parser.add_argument("--residuals", type=Path, required=True, help="aware64 manifest.json with its payload files")
    parser.add_argument("--scales", type=Path, required=True, help="rmsnorm_static_nvfp4_scales_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_bundle(args.weights, args.residuals, args.scales, args.output)
    print(f"Bundle ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
