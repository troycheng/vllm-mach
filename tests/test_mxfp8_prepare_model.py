"""Portable byte-copy/atomicity tests using tiny safetensors and mocked identities."""
from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/prepare_model.py"
spec = importlib.util.spec_from_file_location("mach_prepare_model_cpu_test", PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_safetensors(path, tensors):
    header, payload = {}, b""
    for key, (dtype, shape, raw) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload += raw
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MaterializerCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.bf16, self.mx8, self.output = root / "bf16", root / "mx8", root / "output"
        self.bf16.mkdir(); self.mx8.mkdir()
        self.keys = ("model.language_model.layers.3.self_attn.q_proj.weight",
                     "model.language_model.layers.0.mlp.gate_proj.weight",
                     "model.language_model.layers.0.mlp.up_proj.weight")
        self.retained_key = "model.language_model.embed_tokens.weight"
        source, parent, codes, quant = {}, {}, {}, {}
        for i, key in enumerate(self.keys):
            bf = struct.pack("<64H", *([0x3F80 + i] * 64))
            fp = bytes([40 + i]) * 64
            scale = bytes([127 + i]) * 2
            # Exercise the public BF16 language_model.model prefix mapping.
            mapped = key.replace("model.language_model.", "language_model.model.")
            source[mapped] = ("BF16", [2, 32], bf)
            parent[key] = ("F8_E4M3", [2, 32], fp)
            parent[prepare.scale_key(key)] = ("U8", [2, 1], scale)
            quant[key] = {"shape": [2, 32], "source_sha256": sha(bf),
                          "stored_values_sha256": sha(fp), "stored_scales_sha256": sha(scale)}
            if i:
                codes[key] = ("F8_E4M3", [2, 32], bytes([80 + i]) * 64)
        retained_raw = struct.pack("<4H", 0x3F80, 0x3F00, 0x4000, 0x0000)
        source[self.retained_key] = parent[self.retained_key] = ("BF16", [2, 2], retained_raw)
        retained = {self.retained_key: {"shape": [2, 2], "dtype": "BF16", "sha256": sha(retained_raw),
                    "source_key": self.retained_key, "source_dtype": "BF16", "source_sha256": sha(retained_raw)}}
        write_safetensors(self.bf16 / "model.safetensors", source)
        # Two shards exercise index changes when removing a scale tensor.
        first = {key: value for key, value in parent.items() if key.startswith(self.keys[0].removesuffix(".weight"))}
        second = {key: value for key, value in parent.items() if key not in first}
        write_safetensors(self.mx8 / "model-00001-of-00002.safetensors", first)
        write_safetensors(self.mx8 / "model-00002-of-00002.safetensors", second)
        index = {"metadata": {"total_size": sum(len(v[2]) for v in parent.values())},
                 "weight_map": {key: "model-00001-of-00002.safetensors" if key in first
                                else "model-00002-of-00002.safetensors" for key in parent}}
        (self.mx8 / "model.safetensors.index.json").write_text(json.dumps(index))
        self.codes = root / "replacement_weights.safetensors"
        write_safetensors(self.codes, codes)
        ignores = [f"model.layers.{i}.linear_attn.in_proj_{p}" for i in range(32)
                   if i not in prepare.LAYERS for p in ("a", "b")] + ["lm_head"]
        quantization = copy.deepcopy(prepare._MXFP8_QUANTIZATION_CONFIG)
        quantization["ignore"] = ignores
        config = {"extra": "preserve", "quantization_config": quantization}
        (self.mx8 / "config.json").write_text(json.dumps(config))
        (self.bf16 / "config.json").write_text(json.dumps({"extra": "source"}))
        assets = {"tokenizer.json": "{}", "tokenizer_config.json": '{"chat_template":"fixed"}',
                  "chat_template.jinja": "{{ messages }}", "generation_config.json": "{}"}
        hashes = {}
        for name, text in assets.items():
            (self.mx8 / name).write_text(text)
            hashes[name] = sha(text.encode())
        (self.mx8 / "old-private-audit.json").write_text('{"unrelated":"must not copy"}')
        replacements = {key: sha(raw) for key, (_, _, raw) in codes.items()}
        for name, value in (("QUANTIZED", quant), ("RETAINED", retained), ("L0_REPLACEMENTS", replacements),
                            ("HF_ASSET_HASHES", hashes), ("L0_PAYLOAD_BYTES", self.codes.stat().st_size),
                            ("L0_PAYLOAD_SHA256", prepare.file_sha256(self.codes))):
            patcher = patch.object(prepare, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(prepare, "selected_keys", return_value={self.keys[0]})
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(prepare, "_check_geometry")
        patcher.start(); self.addCleanup(patcher.stop)

    def build(self):
        return prepare.prepare_model(self.bf16, self.mx8, self.codes, self.output)

    def test_full_materialization_preserves_scales_and_copies_assets(self):
        original = {str(path.relative_to(self.mx8)): prepare.file_sha256(path)
                    for path in self.mx8.iterdir()}
        manifest = self.build()
        metadata = prepare.validate_model(self.output, verify_weights=True)
        self.assertTrue(metadata["verified_weights"])
        self.assertEqual(metadata["profile"], prepare.PROFILE)
        _, parent, _ = prepare.scan_checkpoint(self.mx8)
        config, actual, _ = prepare.scan_checkpoint(self.output)
        self.assertEqual(actual[self.keys[0]]["dtype"], "BF16")
        self.assertNotIn(prepare.scale_key(self.keys[0]), actual)
        self.assertEqual(config["extra"], "preserve")
        for key in self.keys[1:]:
            self.assertEqual(prepare.tensor_sha256(actual[key]), prepare.L0_REPLACEMENTS[key])
            scale = prepare.scale_key(key)
            self.assertEqual(prepare.tensor_sha256(actual[scale]), prepare.tensor_sha256(parent[scale]))
        for name in prepare.HF_ASSET_HASHES:
            self.assertEqual((self.mx8 / name).read_bytes(), (self.output / name).read_bytes())
        self.assertFalse((self.output / "old-private-audit.json").exists())
        self.assertEqual(original, {str(path.relative_to(self.mx8)): prepare.file_sha256(path)
                                    for path in self.mx8.iterdir()})
        self.assertNotIn(self.temp.name, json.dumps(manifest))
        self.assertEqual(manifest["source_provenance"], "source_pending_verification")
        self.assertEqual(len(actual), 6)

    def test_quantization_semantic_drift_rejected_before_materialization(self):
        path = self.mx8 / "config.json"
        original = json.loads(path.read_text())
        changes = [("targets", ["Conv2d"]), ("weights.dynamic", True),
                   ("weights.symmetric", False), ("input_activations.dynamic", False),
                   ("input_activations.symmetric", False)]
        for field, value in changes:
            with self.subTest(field=field):
                config = copy.deepcopy(original)
                group = config["quantization_config"]["config_groups"]["group_0"]
                owner = group
                parts = field.split(".")
                for name in parts[:-1]:
                    owner = owner[name]
                owner[parts[-1]] = value
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "quantization configuration differs"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_output_manifest_cannot_certify_changed_quantization_targets(self):
        self.build()
        path = self.output / "config.json"
        config = json.loads(path.read_text())
        config["quantization_config"]["config_groups"]["group_0"]["targets"] = ["Conv2d"]
        path.write_text(json.dumps(config))
        manifest_path = self.output / prepare.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["output_files"]:
            if row["name"] == "config.json":
                row.update(sha256=prepare.file_sha256(path), bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "quantization configuration differs"):
            prepare.validate_model(self.output, verify_weights=True)

    def test_existing_output_and_tensor_mismatch_never_publish(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(sentinel.read_text(), "keep")
        sentinel.unlink(); self.output.rmdir()
        with patch.object(prepare, "L0_PAYLOAD_SHA256", "0" * 64), self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(".output.building-*")))

    def test_failure_cleans_only_the_new_stage(self):
        unrelated = self.output.parent / ".output.building-unrelated"
        unrelated.mkdir()
        (unrelated / "keep").write_text("existing")
        with patch.object(prepare, "_copy_shard", side_effect=RuntimeError("copy failed")), \
             self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".output.building-*")), [unrelated])
        self.assertEqual((unrelated / "keep").read_text(), "existing")

    def test_racing_output_directory_is_not_replaced(self):
        publish = prepare._publish
        def race(stage, output):
            output.mkdir()
            publish(stage, output)
        with patch.object(prepare, "_publish", side_effect=race), self.assertRaises(OSError):
            self.build()
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertFalse(list(self.output.parent.glob(".output.building-*")))

    def test_static_manifest_binding_and_full_payload_validation(self):
        self.build()
        self.assertFalse(prepare.validate_model(self.output)["verified_weights"])
        _, tensors, _ = prepare.scan_checkpoint(self.output)
        item = tensors[self.keys[1]]
        with item["path"].open("r+b") as stream:
            stream.seek(item["base"] + item["data_offsets"][0]); stream.write(b"\xff")
        # The cheap mode certifies only headers and small-file bindings.
        prepare.validate_model(self.output, verify_weights=False)
        with self.assertRaisesRegex(ValueError, "tensor bytes differ"):
            prepare.validate_model(self.output, verify_weights=True)
        profile = self.output / "mach_profile.json"
        value = prepare.read_json(profile); value["kv_scales"][0]["k_scale"] = 1.
        profile.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "KV scales differ"):
            prepare.validate_model(self.output)

    def test_key_mapping_rejects_missing_or_ambiguous_aliases(self):
        key = "model.language_model.layers.0.mlp.gate_proj.weight"
        mapped = "language_model.model.layers.0.mlp.gate_proj.weight"
        self.assertEqual(prepare.resolve_bf16_key(key, {mapped: {}}), mapped)
        with self.assertRaises(ValueError):
            prepare.resolve_bf16_key(key, {})
        with self.assertRaises(ValueError):
            prepare.resolve_bf16_key(key, {mapped: {}, key: {}})


class FixedQuantizationCPU(unittest.TestCase):
    def test_actual_accepted_reconstruction_config_remains_valid(self):
        source = PATH.with_name("reconstruct_model.py").read_text()
        assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "MXFP8_CONFIG_SOURCE"
                                  for target in node.targets))
        config = json.loads(ast.literal_eval(assignment.value))
        prepare._check_geometry(config)
        output = prepare._prepare_config(config)
        self.assertEqual(output["quantization_config"]["config_groups"], config["quantization_config"]["config_groups"])
        # JSON spelling/key ordering is immaterial, but field omission, additions
        # and bool/int substitutions cannot weaken the fixed scheme.
        for mutate in (lambda group: group["weights"].pop("symmetric"),
                       lambda group: group["weights"].update(symmetric=1),
                       lambda group: group.update(output_activations={"dynamic": True})):
            changed = copy.deepcopy(config)
            mutate(changed["quantization_config"]["config_groups"]["group_0"])
            with self.assertRaisesRegex(ValueError, "quantization configuration differs"):
                prepare._prepare_config(changed)


class HeaderCPU(unittest.TestCase):
    def test_shape_size_and_overlapping_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            write_safetensors(path, {"x": ("U8", [2], b"a")})
            with self.assertRaises(ValueError):
                prepare.scan_safetensors(path)
            header = {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                      "b": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"a")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                prepare.scan_safetensors(path)


if __name__ == "__main__":
    unittest.main()
