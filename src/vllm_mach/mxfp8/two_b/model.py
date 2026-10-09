# SPDX-License-Identifier: Apache-2.0
"""CPU reconstruction and validation of the fixed Qwen3.5-2B champion.

No quantized input checkpoint, calibration asset, or GPU is required. The
embedded release manifest binds every output tensor and the serialization.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

from .. import prepare_model as p
from ..reconstruct_model import _blocks, _raw, convert_norm, quantize_block

IDENTITY_PATH = Path(__file__).with_name("data") / "qwen35-2b-model.json"
RECEIPT_NAME = "mach_2b_model.json"


def load_identity():
    return p.read_json(IDENTITY_PATH)


def restored_keys():
    keys = {f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
            for layer in range(3, 24, 4) for proj in ("q_proj", "k_proj", "v_proj", "o_proj")}
    keys.update(f"model.language_model.layers.0.{proj}.weight" for proj in
                ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    keys.update(f"model.language_model.layers.23.mlp.{proj}.weight"
                for proj in ("gate_proj", "up_proj", "down_proj"))
    return keys


def _check_source(bf16, identity):
    files = p._source_tree(bf16)
    source = identity["source"]
    for name, row in source["assets"].items():
        path = bf16 / p._safe_relative(name)
        if not path.is_file() or path.stat().st_size != row["bytes"] or p.file_sha256(path) != row["sha256"]:
            raise ValueError(f"Fixed BF16 source asset differs: {name}")
    shards = [path for path in files if path.suffix == ".safetensors"]
    actual = sorted((path.stat().st_size, p.file_sha256(path)) for path in shards)
    expected = sorted((row["bytes"], row["sha256"]) for row in source["shards"])
    if actual != expected:
        raise ValueError("BF16 source shards differ from the fixed public revision")
    _, tensors, _ = p.scan_checkpoint(bf16)
    return tensors


def _write_model(path, source, identity):
    """Write the frozen header, streaming bounded rows into its exact offsets."""
    import torch
    header = base64.b64decode(identity["header_base64"], validate=True)
    rows = identity["tensors"]
    parsed = json.loads(header, object_pairs_hook=p._unique_object)
    if set(parsed) - {"__metadata__"} != set(rows):
        raise ValueError("Embedded tensor/header inventory differs")
    base = 8 + len(header)
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    quantized = [key for key, row in rows.items() if row["dtype"] in ("F8_E4M3", "F8_E4M3FN")]
    scale_keys = {p.scale_key(key) for key in quantized}
    if any(key not in rows for key in scale_keys):
        raise ValueError("Quantized projection lacks its scale")
    with path.open("x+b") as stream:
        stream.write(struct.pack("<Q", len(header))); stream.write(header)
        for key in quantized:
            item = source[p.resolve_bf16_key(rows[key].get("source_key", key), source)]
            if item["shape"] != rows[key]["shape"]:
                raise ValueError(f"BF16 projection shape differs: {key}")
            offsets = {key: parsed[key]["data_offsets"][0],
                       p.scale_key(key): parsed[p.scale_key(key)]["data_offsets"][0]}
            digests = {name: hashlib.sha256() for name in offsets}
            for x in _blocks(item):
                codes, scales = quantize_block(x)
                for name, tensor in ((key, codes), (p.scale_key(key), scales)):
                    raw = _raw(tensor)
                    digests[name].update(raw)
                    stream.seek(base + offsets[name]); stream.write(raw)
                    offsets[name] += len(raw)
            for name, digest in digests.items():
                if digest.hexdigest() != rows[name]["sha256"]:
                    raise ValueError(f"Reconstructed quantized tensor differs: {name}")
        for key, row in rows.items():
            if key in quantized or key in scale_keys:
                continue
            item = source[p.resolve_bf16_key(row.get("source_key", key), source)]
            if item["shape"] != row["shape"] or row["dtype"] not in dtypes:
                raise ValueError(f"Retained BF16/FP32 geometry differs: {key}")
            offset = parsed[key]["data_offsets"][0]
            digest = hashlib.sha256()
            for x in _blocks(item):
                # Restored projections bypass conversion; retained norms follow
                # the original compressor's BF16 rounding, including RMS +1/-1.
                converted = x.to(dtypes[row["dtype"]]) if key in restored_keys() else convert_norm(key, x, dtypes[row["dtype"]])
                raw = _raw(converted); digest.update(raw)
                stream.seek(base + offset); stream.write(raw); offset += len(raw)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Reconstructed retained tensor differs: {key}")
    if path.stat().st_size != identity["checkpoint"]["bytes"] or p.file_sha256(path) != identity["checkpoint"]["sha256"]:
        raise ValueError("Reconstructed checkpoint serialization differs")


def validate_model(path, *, verify_weights=True):
    """Bind model metadata and optionally hash every weight against this release."""
    path = Path(path).expanduser().absolute()
    p._source_tree(path)
    identity = load_identity()
    config, tensors, index = p.scan_checkpoint(path)
    if index is not None or config != identity["config"]:
        raise ValueError("Fixed 2B config or single-shard contract differs")
    if set(tensors) != set(identity["tensors"]):
        raise ValueError("Fixed 2B tensor inventory differs")
    for key, row in identity["tensors"].items():
        item = tensors[key]
        if item["shape"] != row["shape"] or item["dtype"] != row["dtype"]:
            raise ValueError(f"Fixed 2B tensor header differs: {key}")
        if verify_weights and p.tensor_sha256(item) != row["sha256"]:
            raise ValueError(f"Fixed 2B tensor bytes differ: {key}")
    shard = path / "model.safetensors"
    if shard.stat().st_size != identity["checkpoint"]["bytes"]:
        raise ValueError("Fixed 2B checkpoint size differs")
    if verify_weights and p.file_sha256(shard) != identity["checkpoint"]["sha256"]:
        raise ValueError("Fixed 2B checkpoint hash differs")
    for name, row in identity["assets"].items():
        asset = path / p._safe_relative(name)
        if not asset.is_file() or asset.stat().st_size != row["bytes"] or p.file_sha256(asset) != row["sha256"]:
            raise ValueError(f"Fixed 2B tokenizer/config asset differs: {name}")
    return {"verified_weights": bool(verify_weights), "tensor_count": len(tensors),
            "checkpoint_sha256": identity["checkpoint"]["sha256"],
            "bf16_repository": identity["source"]["repository"],
            "bf16_revision": identity["source"]["revision"],
            "restored_bf16_weights": sorted(restored_keys())}


def reconstruct_model(bf16, output):
    """Rebuild from the pinned public BF16 snapshot and publish without overwrite."""
    bf16, output = (Path(value).expanduser().absolute() for value in (bf16, output))
    if bf16.is_symlink() or output.is_symlink():
        raise ValueError("Checkpoint directories must not be symlinks")
    bf16, output = bf16.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.is_relative_to(bf16) or bf16.is_relative_to(output):
        raise ValueError("Output must have an existing parent and be independent of BF16")
    identity = load_identity()
    source = _check_source(bf16, identity)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    published = False
    try:
        _write_model(stage / "model.safetensors", source, identity)
        (stage / "config.json").write_text(identity["config_text"])
        for name in identity["assets"]:
            if name != "config.json":
                shutil.copy2(bf16 / name, stage / name)
        result = validate_model(stage, verify_weights=True)
        result.update(status="complete", operation="CPU BF16 reconstruction",
                      model_identity_sha256=p.file_sha256(IDENTITY_PATH))
        (stage / RECEIPT_NAME).write_text(json.dumps(result, indent=2) + "\n")
        p._publish(stage, output)
        published = True
        return result
    finally:
        if not published:
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct_model(args.bf16, args.output), indent=2))


if __name__ == "__main__":
    main()
