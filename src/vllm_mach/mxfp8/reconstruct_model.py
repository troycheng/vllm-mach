# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the fixed champion from BF16 and the independent L0 code asset.

CPU only, bounded tensor chunks, no original MXFP8 checkpoint or audit inputs.
Embedded identities bind bytes; upstream revision and asset rights remain pending.
"""
from __future__ import annotations

import copy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

from . import prepare_model as p

# The fixed source metadata is converted to a text-only MXFP8 configuration.
# These are public-format model metadata, not calibration or acquisition records.
BF16_METADATA_HASHES = {'config.json': 'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670', 'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8', 'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'}
MXFP8_CONFIG_SOURCE = '{\n  "architectures": [\n    "Qwen3_5ForCausalLM"\n  ],\n  "attention_bias": false,\n  "attention_dropout": 0.0,\n  "attn_output_gate": true,\n  "bos_token_id": null,\n  "dtype": "bfloat16",\n  "eos_token_id": 248044,\n  "full_attention_interval": 4,\n  "head_dim": 256,\n  "hidden_act": "silu",\n  "hidden_size": 2560,\n  "initializer_range": 0.02,\n  "intermediate_size": 9216,\n  "layer_types": [\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention",\n    "linear_attention",\n    "linear_attention",\n    "linear_attention",\n    "full_attention"\n  ],\n  "linear_conv_kernel_dim": 4,\n  "linear_key_head_dim": 128,\n  "linear_num_key_heads": 16,\n  "linear_num_value_heads": 32,\n  "linear_value_head_dim": 128,\n  "mamba_ssm_dtype": "float32",\n  "max_position_embeddings": 262144,\n  "mlp_only_layers": [],\n  "model_type": "qwen3_5_text",\n  "mtp_num_hidden_layers": 1,\n  "mtp_use_dedicated_embeddings": false,\n  "num_attention_heads": 16,\n  "num_hidden_layers": 32,\n  "num_key_value_heads": 4,\n  "pad_token_id": null,\n  "partial_rotary_factor": 0.25,\n  "quantization_config": {\n    "config_groups": {\n      "group_0": {\n        "format": "mxfp8-quantized",\n        "input_activations": {\n          "actorder": null,\n          "block_structure": null,\n          "dynamic": true,\n          "group_size": 32,\n          "num_bits": 8,\n          "observer": null,\n          "observer_kwargs": {},\n          "scale_dtype": "torch.uint8",\n          "strategy": "group",\n          "symmetric": true,\n          "type": "float",\n          "zp_dtype": null\n        },\n        "output_activations": null,\n        "targets": [\n          "Linear"\n        ],\n        "weights": {\n          "actorder": null,\n          "block_structure": null,\n          "dynamic": false,\n          "group_size": 32,\n          "num_bits": 8,\n          "observer": "memoryless_minmax",\n          "observer_kwargs": {},\n          "scale_dtype": "torch.uint8",\n          "strategy": "group",\n          "symmetric": true,\n          "type": "float",\n          "zp_dtype": null\n        }\n      }\n    },\n    "format": "mxfp8-quantized",\n    "global_compression_ratio": null,\n    "ignore": [\n      "model.layers.0.linear_attn.in_proj_b",\n      "model.layers.0.linear_attn.in_proj_a",\n      "model.layers.1.linear_attn.in_proj_b",\n      "model.layers.1.linear_attn.in_proj_a",\n      "model.layers.2.linear_attn.in_proj_b",\n      "model.layers.2.linear_attn.in_proj_a",\n      "model.layers.4.linear_attn.in_proj_b",\n      "model.layers.4.linear_attn.in_proj_a",\n      "model.layers.5.linear_attn.in_proj_b",\n      "model.layers.5.linear_attn.in_proj_a",\n      "model.layers.6.linear_attn.in_proj_b",\n      "model.layers.6.linear_attn.in_proj_a",\n      "model.layers.8.linear_attn.in_proj_b",\n      "model.layers.8.linear_attn.in_proj_a",\n      "model.layers.9.linear_attn.in_proj_b",\n      "model.layers.9.linear_attn.in_proj_a",\n      "model.layers.10.linear_attn.in_proj_b",\n      "model.layers.10.linear_attn.in_proj_a",\n      "model.layers.12.linear_attn.in_proj_b",\n      "model.layers.12.linear_attn.in_proj_a",\n      "model.layers.13.linear_attn.in_proj_b",\n      "model.layers.13.linear_attn.in_proj_a",\n      "model.layers.14.linear_attn.in_proj_b",\n      "model.layers.14.linear_attn.in_proj_a",\n      "model.layers.16.linear_attn.in_proj_b",\n      "model.layers.16.linear_attn.in_proj_a",\n      "model.layers.17.linear_attn.in_proj_b",\n      "model.layers.17.linear_attn.in_proj_a",\n      "model.layers.18.linear_attn.in_proj_b",\n      "model.layers.18.linear_attn.in_proj_a",\n      "model.layers.20.linear_attn.in_proj_b",\n      "model.layers.20.linear_attn.in_proj_a",\n      "model.layers.21.linear_attn.in_proj_b",\n      "model.layers.21.linear_attn.in_proj_a",\n      "model.layers.22.linear_attn.in_proj_b",\n      "model.layers.22.linear_attn.in_proj_a",\n      "model.layers.24.linear_attn.in_proj_b",\n      "model.layers.24.linear_attn.in_proj_a",\n      "model.layers.25.linear_attn.in_proj_b",\n      "model.layers.25.linear_attn.in_proj_a",\n      "model.layers.26.linear_attn.in_proj_b",\n      "model.layers.26.linear_attn.in_proj_a",\n      "model.layers.28.linear_attn.in_proj_b",\n      "model.layers.28.linear_attn.in_proj_a",\n      "model.layers.29.linear_attn.in_proj_b",\n      "model.layers.29.linear_attn.in_proj_a",\n      "model.layers.30.linear_attn.in_proj_b",\n      "model.layers.30.linear_attn.in_proj_a",\n      "lm_head"\n    ],\n    "kv_cache_scheme": null,\n    "quant_method": "compressed-tensors",\n    "quantization_status": "compressed",\n    "sparsity_config": {},\n    "transform_config": {},\n    "version": "0.17.0"\n  },\n  "rms_norm_eps": 1e-06,\n  "rope_parameters": {\n    "mrope_interleaved": true,\n    "mrope_section": [\n      11,\n      11,\n      10\n    ],\n    "partial_rotary_factor": 0.25,\n    "rope_theta": 10000000,\n    "rope_type": "default"\n  },\n  "tie_word_embeddings": true,\n  "transformers_version": "5.17.0",\n  "use_cache": true,\n  "vocab_size": 248320\n}'
GENERATED_ASSETS = {'tokenizer_config.json': '{\n  "add_prefix_space": false,\n  "audio_bos_token": "<|audio_start|>",\n  "audio_eos_token": "<|audio_end|>",\n  "audio_token": "<|audio_pad|>",\n  "backend": "tokenizers",\n  "bos_token": null,\n  "clean_up_tokenization_spaces": false,\n  "eos_token": "<|im_end|>",\n  "errors": "replace",\n  "image_token": "<|image_pad|>",\n  "is_local": true,\n  "local_files_only": false,\n  "model_max_length": 262144,\n  "model_specific_special_tokens": {\n    "audio_bos_token": "<|audio_start|>",\n    "audio_eos_token": "<|audio_end|>",\n    "audio_token": "<|audio_pad|>",\n    "image_token": "<|image_pad|>",\n    "video_token": "<|video_pad|>",\n    "vision_bos_token": "<|vision_start|>",\n    "vision_eos_token": "<|vision_end|>"\n  },\n  "pad_token": "<|endoftext|>",\n  "pretokenize_regex": "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+",\n  "split_special_tokens": false,\n  "tokenizer_class": "Qwen2Tokenizer",\n  "unk_token": null,\n  "video_token": "<|video_pad|>",\n  "vision_bos_token": "<|vision_start|>",\n  "vision_eos_token": "<|vision_end|>"\n}\n', 'generation_config.json': '{\n  "_from_model_config": true,\n  "eos_token_id": 248044,\n  "transformers_version": "5.17.0",\n  "use_cache": true\n}\n'}

CHUNK_BYTES = 4 * 2**20
BF16_REPOSITORY = 'Qwen/Qwen3.5-4B'
BF16_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BF16_TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
BF16_SHARD_IDENTITIES = {'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188': 3990429408, '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61': 5329398688}


def quantize_block(x):
    """Exact audited BF16 group32 E4M3/E8M0 conversion, on CPU.

    Preserve BF16 maximum and scale arithmetic. The mantissa mask includes
    the fixed 32-bit increment in BF16 encoding, as in the audited observer.
    Zero/subnormal handling follows the same log2 and uint8 conversions.
    """
    import torch
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 32:
        raise ValueError("Quantizer requires a CPU BF16 matrix with group32 columns")
    maximum = x.reshape(x.shape[0], -1, 32).abs().amax(-1)
    bits = maximum.view(torch.uint16).to(torch.int32)
    rounded = ((bits + 32) & 0xFF80).to(torch.uint16).view(torch.bfloat16)
    scales = (127 + torch.floor(torch.log2(rounded)) - 8).to(torch.uint8)
    expanded = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    codes = (x.float() / expanded).to(torch.float8_e4m3fn)
    return codes, scales


def norm_recipe(key):
    if key.endswith(".linear_attn.norm.weight"):
        return "bf16"
    if (key == "model.language_model.norm.weight"
            or key.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight",
                             ".self_attn.q_norm.weight", ".self_attn.k_norm.weight"))):
        return "float32_plus1_bf16_minus1"
    return None


def convert_norm(key, x, dtype):
    import torch
    recipe = norm_recipe(key)
    if recipe == "bf16":
        return x.to(torch.bfloat16).to(dtype)
    if recipe == "float32_plus1_bf16_minus1":
        return ((x.float() + 1).to(torch.bfloat16).float() - 1).to(dtype)
    return x.to(dtype)


def _blocks(item):
    """Read whole rows, retaining at most CHUNK_BYTES of source tensor bytes."""
    import torch
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    if item["dtype"] not in dtypes:
        raise ValueError("Reconstruction source must be BF16 or FP32")
    width = item["shape"][-1] if item["shape"] else 1
    row_bytes = width * p._DTYPE_BYTES[item["dtype"]]
    rows = max(1, CHUNK_BYTES // row_bytes)
    begin, end = item["data_offsets"]
    with item["path"].open("rb") as stream:
        stream.seek(item["base"] + begin)
        remaining = end - begin
        while remaining:
            length = min(remaining, rows * row_bytes)
            raw = bytearray(stream.read(length))
            if len(raw) != length:
                raise ValueError("Truncated BF16 source during reconstruction")
            yield torch.frombuffer(raw, dtype=dtypes[item["dtype"]]).reshape(-1, width)
            remaining -= length


def _raw(x):
    # uint8 view works on CPU even for BF16/FP8, without dtype-dependent NumPy.
    import torch
    return x.contiguous().view(torch.uint8).numpy().tobytes()


def _copy_tensor(stream, item, offset):
    stream.seek(offset)
    begin, end = item["data_offsets"]
    for chunk in p._regions(item["path"], item["base"] + begin, end - begin):
        stream.write(chunk)


def _prepare_tokenizer(bf16, stage):
    """Use standard Transformers conversion, then enforce exact artifact hashes."""
    from transformers import AutoTokenizer
    temporary = Path(tempfile.mkdtemp(prefix=".tokenizer-", dir=stage))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(bf16), local_files_only=True)
        tokenizer.save_pretrained(str(temporary))
        for name in ("tokenizer.json", "chat_template.jinja"):
            if not (temporary / name).is_file() or p.file_sha256(temporary / name) != p.HF_ASSET_HASHES[name]:
                raise ValueError(f"Standard tokenizer conversion hash differs: {name}")
            shutil.copy2(temporary / name, stage / name)
        actual = p.read_json(temporary / "tokenizer_config.json")
        expected = json.loads(GENERATED_ASSETS["tokenizer_config.json"])
        # local_files_only reflects this offline load, not tokenizer semantics.
        actual.pop("local_files_only", None); expected.pop("local_files_only", None)
        if actual != expected:
            raise ValueError("Standard tokenizer configuration conversion differs")
        return {"method": "AutoTokenizer.from_pretrained(local_files_only=True).save_pretrained",
                "transformers": version("transformers"), "tokenizers": version("tokenizers")}
    finally:
        shutil.rmtree(temporary)


def _write_model(path, source, replacements):
    """Write one shard, while independently checking all original MXFP8 bytes."""
    expected = p._expected_output_tensors()
    header, offsets, position = {}, {}, 0
    for key, row in sorted(expected.items()):
        dtype = "F8_E4M3" if row["dtype"] == "E4M3" else row["dtype"]
        size = p._DTYPE_BYTES[dtype] * math.prod(row["shape"])
        offsets[key] = position
        header[key] = {"dtype": dtype, "shape": row["shape"], "data_offsets": [position, position + size]}
        position += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    base = 8 + len(encoded)
    restored = p.selected_keys()
    with path.open("x+b") as stream:
        stream.write(struct.pack("<Q", len(encoded))); stream.write(encoded)
        for key, row in p.QUANTIZED.items():
            item = source[p.resolve_bf16_key(key, source)]
            code_hash, scale_hash = hashlib.sha256(), hashlib.sha256()
            code_position = scale_position = 0
            for x in _blocks(item):
                codes, scales = quantize_block(x)
                raw_codes, raw_scales = _raw(codes), _raw(scales)
                code_hash.update(raw_codes); scale_hash.update(raw_scales)
                if key not in restored:
                    if key not in p.L0_REPLACEMENTS:
                        stream.seek(base + offsets[key] + code_position); stream.write(raw_codes)
                    stream.seek(base + offsets[p.scale_key(key)] + scale_position); stream.write(raw_scales)
                code_position += len(raw_codes); scale_position += len(raw_scales)
            if (code_hash.hexdigest() != row["stored_values_sha256"]
                    or scale_hash.hexdigest() != row["stored_scales_sha256"]):
                raise ValueError(f"Original MXFP8 reconstruction hash differs: {key}")
            if key in restored:
                _copy_tensor(stream, item, base + offsets[key])
            elif key in p.L0_REPLACEMENTS:
                _copy_tensor(stream, replacements[key], base + offsets[key])
        for key, row in p.RETAINED.items():
            item = source[p.resolve_bf16_key(row["source_key"], source)]
            if norm_recipe(key) is None and row["dtype"] == item["dtype"]:
                if row["dtype"] != item["dtype"] or row["sha256"] != p.tensor_sha256(item):
                    raise ValueError(f"No exact retained-tensor reconstruction rule: {key}")
                _copy_tensor(stream, item, base + offsets[key])
            else:
                import torch
                dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[row["dtype"]]
                digest, position_here = hashlib.sha256(), 0
                for x in _blocks(item):
                    raw = _raw(convert_norm(key, x, dtype))
                    digest.update(raw)
                    stream.seek(base + offsets[key] + position_here); stream.write(raw)
                    position_here += len(raw)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"Retained norm reconstruction hash differs: {key}")
    actual = p.scan_safetensors(path)
    for key, row in expected.items():
        p._require_identity(key, actual[key], row["shape"], row["dtype"], row["sha256"])
    return position


def reconstruct_model(bf16, l0_codes, output):
    """Reconstruct and atomically publish the fixed 595-tensor champion asset."""
    paths = [Path(value).expanduser().absolute() for value in (bf16, l0_codes, output)]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Checkpoint inputs and output must not be symlinks")
    bf16, l0_codes, output = (path.resolve() for path in paths)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.is_relative_to(bf16) or bf16.is_relative_to(output):
        raise ValueError("Output must have an existing parent and be independent of BF16")
    source_files = p._source_tree(bf16)
    if l0_codes.is_dir():
        l0_codes = l0_codes / "replacement_weights.safetensors"
    if l0_codes.is_symlink() or not l0_codes.is_file():
        raise ValueError("L0 codes must be a real safetensors file")
    if l0_codes.stat().st_size != p.L0_PAYLOAD_BYTES or p.file_sha256(l0_codes) != p.L0_PAYLOAD_SHA256:
        raise ValueError("Fixed L0 safetensors payload size/hash differs")
    replacement = p.scan_safetensors(l0_codes)
    if set(replacement) != set(p.L0_REPLACEMENTS):
        raise ValueError("L0 codes must contain exactly gate_proj and up_proj")
    config, source, _ = p.scan_checkpoint(bf16)
    p._check_geometry(config)
    # Validate the fixed source metadata before converting it to text-only form.
    for name, digest in BF16_METADATA_HASHES.items():
        if not (bf16 / name).is_file() or p.file_sha256(bf16 / name) != digest:
            raise ValueError(f"Fixed BF16 metadata differs: {name}")
    if not (bf16 / "tokenizer.json").is_file() or p.file_sha256(bf16 / "tokenizer.json") != BF16_TOKENIZER_SHA256:
        raise ValueError("BF16 tokenizer payload is not the fixed release asset")
    new_config = p._prepare_config(json.loads(MXFP8_CONFIG_SOURCE))
    for key, row in p.QUANTIZED.items():
        mapped = p.resolve_bf16_key(key, source)
        p._require_identity(mapped, source[mapped], row["shape"], "BF16", row["source_sha256"])
    for key, row in p.RETAINED.items():
        mapped = p.resolve_bf16_key(row["source_key"], source)
        p._require_identity(mapped, source[mapped], row["shape"], row["source_dtype"], row["source_sha256"])
    for key, digest in p.L0_REPLACEMENTS.items():
        p._require_identity(key, replacement[key], p.QUANTIZED[key]["shape"], "E4M3", digest)
    expected = p._expected_output_tensors()
    payload_bytes = sum(math.prod(row["shape"]) * p._DTYPE_BYTES[
        "F8_E4M3" if row["dtype"] == "E4M3" else row["dtype"]] for row in expected.values())
    asset_bytes = sum((bf16 / name).stat().st_size for name in ("tokenizer.json", "chat_template.jinja"))
    asset_bytes += sum(len(text.encode()) for text in GENERATED_ASSETS.values())
    if shutil.disk_usage(output.parent).free < payload_bytes + 4 * asset_bytes + 4 * 2**20:
        raise OSError("Insufficient disk space for reconstructed checkpoint")
    inputs = {"bf16": {"provenance": "public_hf_revision_bytes_verified",
                       "repository": BF16_REPOSITORY, "revision": BF16_REVISION, "files": [
        {"name": str(path.relative_to(bf16)), "sha256": p.file_sha256(path), "bytes": path.stat().st_size}
        for path in source_files if path.suffix == ".safetensors" or path.name in BF16_METADATA_HASHES
        or path.name == "tokenizer.json" or path.name.endswith(".safetensors.index.json")]},
        "l0_codes": {"provenance": "source_pending_verification", "format": "safetensors",
                     "sha256": p.L0_PAYLOAD_SHA256, "bytes": p.L0_PAYLOAD_BYTES}}
    shard_identities = {row["sha256"]: row["bytes"] for row in inputs["bf16"]["files"]
                        if row["name"].endswith(".safetensors")}
    if shard_identities != BF16_SHARD_IDENTITIES:
        raise ValueError("BF16 tensor shards differ from the fixed public revision")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    published = False
    try:
        tokenizer_conversion = _prepare_tokenizer(bf16, stage)
        _write_model(stage / "model.safetensors", source, replacement)
        p._write_json(stage / "config.json", new_config)
        for name, text in GENERATED_ASSETS.items():
            with (stage / name).open("xb") as stream:
                stream.write(text.encode())
        p._write_json(stage / "mach_profile.json", {"profile": p.PROFILE, "kv_scales": copy.deepcopy(p.KV_SCALES)})
        _, actual, _ = p.scan_checkpoint(stage)
        tensors = []
        restored = p.selected_keys()
        for key, item in sorted(actual.items()):
            kind = "restored_bf16" if key in restored else "asymmetric_l0_code" if key in p.L0_REPLACEMENTS else "reconstructed_original_mxfp8"
            tensors.append({"key": key, "dtype": item["dtype"], "shape": item["shape"],
                            "sha256": expected[key]["sha256"], "kind": kind})
        copied = [{"name": path.name, "kind": "reconstructed" if path.suffix == ".safetensors" else "fixed_metadata",
                   "sha256": p.file_sha256(path), "bytes": path.stat().st_size} for path in sorted(stage.iterdir())]
        manifest = {"status": "complete", "profile": p.PROFILE,
                    "scope": "checkpoint byte reconstruction; loader and GPU fidelity unqualified",
                    "source_provenance": "source_pending_verification", "inputs": inputs,
                    "output_files": copied, "output_tensors": tensors,
                    "restored_bf16_weights": sorted(restored),
                    "removed_scale_keys": sorted(p.scale_key(key) for key in restored),
                    "l0_replacement_keys": sorted(p.L0_REPLACEMENTS),
                    "unmodified_tensor_bytes_verified": True, "tensor_payload_bytes": payload_bytes,
                    "original_mxfp8_tensor_bytes_verified": True,
                    "tokenizer_conversion": tokenizer_conversion,
                    "reconstruction": {"group_size": 32, "chunk_source_bytes": CHUNK_BYTES,
                                       "quantized_tensors": len(p.QUANTIZED),
                                       "norm_recipes": {recipe: sum(norm_recipe(key) == recipe for key in p.RETAINED)
                                                        for recipe in ("bf16", "float32_plus1_bf16_minus1")}},
                    "materializer_sha256": p.file_sha256(Path(p.__file__)),
                    "reconstructor_sha256": p.file_sha256(Path(__file__))}
        p._write_json(stage / p.MANIFEST_NAME, manifest)
        p.validate_model(stage, verify_weights=True)
        p._publish(stage, output)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(stage)
