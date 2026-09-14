#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the existing QA Hessian packer and QA/code residual fit, then package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, assets, checked_model, module, sha, train_rows, write_json


def read_capture(work, kind, rank):
    import torch
    manifest_path = work / "manifests" / f"{kind}.json"
    manifest = json.loads(manifest_path.read_text())
    completed = json.loads((work / kind / "COMPLETE.json").read_text())
    if completed["manifest_sha256"] != sha(manifest_path):
        raise ValueError("Capture belongs to another calibration manifest")
    records = completed["records"]
    if len(records) != 2 or {r["rank"] for r in records} != {0, 1}:
        raise ValueError("Capture must contain both TP ranks")
    record = next(r for r in records if r["rank"] == rank)
    path = assets.checked_path(work / kind, record["file"], record["sha256"])
    data = torch.load(path, weights_only=True, map_location="cpu")
    if data["rank"] != rank:
        raise ValueError("Capture rank mismatch")
    return data, manifest


def original_gu(model, index, layer, rank, device):
    import torch
    from safetensors import safe_open
    parts = []
    for role in ("gate_proj", "up_proj"):
        name = f"model.language_model.layers.{layer}.mlp.{role}.weight"
        with safe_open(model / index[name], framework="pt", device="cpu") as file:
            parts.append(file.get_slice(name)[rank * 8704:(rank + 1) * 8704, :].contiguous())
    gu = torch.cat(parts).contiguous()
    if gu.dtype != torch.bfloat16 or tuple(gu.shape) != (17408, 5120):
        raise ValueError("Expected original BF16 TP2 gate/up [17408,5120]")
    return gu.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "work-dir", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    model, work, output = args.model.resolve(), args.work_dir.resolve(), args.output.resolve()
    index = checked_model(model)
    if output.exists():
        raise FileExistsError(output)
    import torch
    import offline_quant
    import fit_rank64
    import weight_container
    import derive_static_scales
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    scale_path = work / "static_scales.json"
    write_json(scale_path, derive_static_scales.derive(model))
    assets.validate_static_scales(scale_path)
    weights_dir, residual_dir = work / "weights", work / "residuals"
    for directory in (weights_dir, residual_dir):
        directory.mkdir(exist_ok=False)
    weights = {"schema": 1, "mask": list(assets.MASK),
               "official_config_sha256": assets.OFFICIAL_CONFIG,
               "official_index_sha256": assets.OFFICIAL_INDEX, "entries": []}
    residuals = {"schema": 1, "mask": list(assets.MASK), "entries": []}
    fit_records = []
    tsha = assets.tensor_sha256
    for rank in (0, 1):
        qa, qa_manifest = read_capture(work, "qa", rank)
        code, code_manifest = read_capture(work, "code", rank)
        for layer in assets.MASK:
            qa_train = train_rows(qa, qa_manifest["samples"], layer, qa_manifest["decode_offsets"]).to(args.device)
            code_train = train_rows(code, code_manifest["samples"], layer, code_manifest["decode_offsets"]).to(args.device)
            blocks = qa_train.float().reshape(128, 320, 16).permute(1, 0, 2).contiguous()
            hessian = torch.bmm(blocks.transpose(1, 2), blocks) / 128
            gu = original_gu(model, index, layer, rank, args.device)
            weight, recipe = offline_quant.optimize_and_pack(gu, hessian, weight_container)
            origin_hash = tsha(gu)
            payload = {"layer": layer, "rank": rank, "official_bf16_gu_sha256": origin_hash,
                       **{k: getattr(weight, k).cpu() for k in ("packed", "scales", "global_scale")}}
            name = f"layer{layer:02d}_rank{rank}.pt"
            with (weights_dir / name).open("xb") as stream:
                torch.save(payload, stream)
            hashes = {k: tsha(payload[k]) for k in ("packed", "scales", "global_scale")}
            weights["entries"].append({"layer": layer, "rank": rank, "file": name,
                "file_sha256": sha(weights_dir / name), "official_bf16_gu_sha256": origin_hash,
                **{k + "_sha256": v for k, v in hashes.items()}})
            delta = gu.float() - offline_quant.dequantize_nv4(weight).float()
            factors, receipt = fit_rank64.fit(delta, torch.cat((qa_train, code_train)).contiguous(), tsha)
            a, b = factors["aware64"]
            with (residual_dir / name).open("xb") as stream:
                torch.save({"layer": layer, "rank": rank, "aware64_a": a.cpu(), "aware64_b": b.cpu()}, stream)
            residuals["entries"].append({"layer": layer, "rank": rank, "file": name,
                "file_sha256": sha(residual_dir / name), "base_file_sha256": sha(weights_dir / name),
                "base_tensor_sha256": hashes, "official_bf16_gu_sha256": origin_hash,
                "tensor_sha256": {"aware64_a": tsha(a), "aware64_b": tsha(b)}})
            fit_records.append({"layer": layer, "rank": rank, "quantizer": recipe, "fit": receipt})
            print(f"Generated layer {layer}, TP rank {rank}", flush=True)
            del blocks, hessian, gu, weight, payload, delta, factors, a, b, qa_train, code_train
        del qa, code
    weights["entries"].sort(key=lambda e: (e["layer"], e["rank"]))
    residuals["entries"].sort(key=lambda e: (e["layer"], e["rank"]))
    write_json(weights_dir / "manifest.json", weights)
    write_json(residual_dir / "manifest.json", residuals)
    write_json(work / "fit-records.json", fit_records)
    importer = module("mach_generated_bundle_import", ROOT / "tools/import_rank64_bundle.py")
    importer.import_bundle(weights_dir / "manifest.json", residual_dir / "manifest.json", scale_path, output)
    write_json(output / "generation.json", {"method": "QA Hessian NVFP4 + mixed QA/code aware64",
               "qa_capture_sha256": sha(work / "qa/COMPLETE.json"),
               "code_capture_sha256": sha(work / "code/COMPLETE.json"),
               "fit_records_sha256": sha(work / "fit-records.json")})


if __name__ == "__main__":
    main()
