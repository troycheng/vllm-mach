"""CPU tests of the 2B reconstruction trust and publication boundaries."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_mach.mxfp8.two_b import model


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class ModelCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"; self.source.mkdir()
        self.output = self.root / "output"
        self.key = "model.language_model.layers.0.mlp.gate_proj.weight"
        self.raw = b"\x00\x3f\x80\x3f"
        self.header = json.dumps({"__metadata__": {"format": "pt"}, self.key:
                                 {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]}}).encode()
        self.header += b" " * (-len(self.header) % 8)
        self.payload = struct.pack("<Q", len(self.header)) + self.header + self.raw
        self.config = {"model_type": "qwen3_5_text", "num_hidden_layers": 24}
        self.config_text = json.dumps(self.config, indent=2) + "\n"
        for name, raw in (("config.json", self.config_text.encode()), ("tokenizer.json", b'{"complete":true}'),
                          ("model.safetensors", self.payload)):
            (self.source / name).write_bytes(raw)
        assets = {name: {"bytes": len((self.source / name).read_bytes()), "sha256": digest((self.source / name).read_bytes())}
                  for name in ("config.json", "tokenizer.json")}
        self.identity = {"config": self.config, "config_text": self.config_text,
                         "header_base64": base64.b64encode(self.header).decode(), "assets": assets,
                         "checkpoint": {"bytes": len(self.payload), "sha256": digest(self.payload)},
                         "tensors": {self.key: {"dtype": "BF16", "shape": [1, 2], "sha256": digest(self.raw)}},
                         "source": {"repository": "Qwen/Qwen3.5-2B", "revision": "test",
                                    "assets": assets, "shards": [{"bytes": len(self.payload), "sha256": digest(self.payload)}]}}
        self.identity_path = self.root / "identity.json"
        self.identity_path.write_text(json.dumps(self.identity))
        patcher = patch.object(model, "IDENTITY_PATH", self.identity_path)
        patcher.start(); self.addCleanup(patcher.stop)

    def build(self):
        # These lifecycle tests avoid Torch. The real writer arithmetic is
        # covered below and by full release reconstruction on CPU.
        with patch.object(model, "_write_model", side_effect=lambda path, *args: path.write_bytes(self.payload)):
            return model.reconstruct_model(self.source, self.output)

    def test_build_validates_and_preserves_complete_tokenizer(self):
        result = self.build()
        self.assertEqual(result["tensor_count"], 1)
        self.assertTrue(result["verified_weights"])
        self.assertEqual((self.output / "tokenizer.json").read_bytes(), (self.source / "tokenizer.json").read_bytes())
        self.assertEqual((self.output / "model.safetensors").read_bytes(), self.payload)

    def test_source_drift_rejected_before_stage_creation(self):
        (self.source / "model.safetensors").write_bytes(self.payload[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "source shards differ"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".output.building-*")), [])

    def test_payload_drift_is_not_certified_by_header_only_validation(self):
        self.build()
        (self.output / "model.safetensors").write_bytes(self.payload[:-1] + b"x")
        self.assertFalse(model.validate_model(self.output, verify_weights=False)["verified_weights"])
        with self.assertRaisesRegex(ValueError, "tensor bytes differ"):
            model.validate_model(self.output)

    def test_config_ignore_drift_and_tokenizer_truncation_are_rejected(self):
        self.build()
        (self.output / "config.json").write_text(json.dumps({**self.config, "quantization_config": {"ignore": []}}))
        with self.assertRaisesRegex(ValueError, "config"):
            model.validate_model(self.output)
        (self.output / "config.json").write_text(self.config_text)
        (self.output / "tokenizer.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "asset differs"):
            model.validate_model(self.output)

    def test_existing_output_never_overwritten_and_failure_cleans_only_own_stage(self):
        unrelated = self.root / ".output.building-unrelated"; unrelated.mkdir()
        with patch.object(model, "_write_model", side_effect=ValueError("quantized tensor differs")):
            with self.assertRaisesRegex(ValueError, "quantized tensor differs"):
                model.reconstruct_model(self.source, self.output)
        self.assertEqual(list(self.root.glob(".output.building-*")), [unrelated])
        self.output.mkdir(); (self.output / "keep").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.output / "keep").read_text(), "keep")

    def test_symlink_source_and_ambiguous_source_alias_rejected(self):
        alias = self.root / "alias"; alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            model.reconstruct_model(alias, self.output)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            model.p.resolve_bf16_key(self.key, {self.key: {}, self.key.replace("model.language_model.", "model."): {}})

    def test_restoration_is_33_exact_projections_without_4b_l0_asset(self):
        keys = model.restored_keys()
        self.assertEqual(len(keys), 33)
        self.assertIn(self.key, keys)
        self.assertIn("model.language_model.layers.23.mlp.down_proj.weight", keys)
        self.assertNotIn("model.language_model.layers.1.mlp.down_proj.weight", keys)

    def test_bundled_release_binds_438_tensors_and_restoration_scales(self):
        identity = model.p.read_json(Path(model.__file__).with_name("data") / "qwen35-2b-model.json")
        tensors = identity["tensors"]
        self.assertEqual(len(tensors), 438)
        self.assertEqual(identity["config"]["num_hidden_layers"], 24)
        self.assertEqual(identity["config"]["hidden_size"], 2048)
        for key in model.restored_keys():
            self.assertEqual(tensors[key]["dtype"], "BF16")
            self.assertNotIn(model.p.scale_key(key), tensors)
        header = json.loads(base64.b64decode(identity["header_base64"]))
        self.assertEqual(set(header) - {"__metadata__"}, set(tensors))
        for key, row in tensors.items():
            self.assertEqual(header[key]["shape"], row["shape"])
            self.assertEqual(header[key]["dtype"], row["dtype"])
        self.assertEqual(tensors["lm_head.weight"]["source_key"], "model.language_model.embed_tokens.weight")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is needed for actual CPU reconstruction arithmetic")
    def test_real_quantizer_norm_and_streaming_writer(self):
        import torch
        x = torch.arange(64, dtype=torch.float32).reshape(2, 32).to(torch.bfloat16) / 16
        q, s = model.quantize_block(x)
        key = "model.language_model.layers.1.mlp.gate_proj.weight"
        norm = "model.language_model.layers.1.input_layernorm.weight"
        source_norm = torch.tensor([0.013, -0.023], dtype=torch.float32)
        norm_out = model.convert_norm(norm, source_norm, torch.bfloat16)
        source_rows = {key: ("BF16", list(x.shape), model._raw(x)), norm: ("F32", [2], model._raw(source_norm))}
        output_rows = {key: ("F8_E4M3", list(q.shape), model._raw(q)), model.p.scale_key(key): ("U8", list(s.shape), model._raw(s)),
                       norm: ("BF16", [2], model._raw(norm_out))}
        def binary(rows):
            header, payload = {"__metadata__": {"format": "pt"}}, b""
            for name, (dtype, shape, raw) in rows.items():
                header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
                payload += raw
            encoded = json.dumps(header).encode(); encoded += b" " * (-len(encoded) % 8)
            return encoded, struct.pack("<Q", len(encoded)) + encoded + payload
        _, source_bytes = binary(source_rows)
        header, output_bytes = binary(output_rows)
        source_file = self.root / "real-source.safetensors"; source_file.write_bytes(source_bytes)
        identity = {"header_base64": base64.b64encode(header).decode(),
                    "checkpoint": {"bytes": len(output_bytes), "sha256": digest(output_bytes)},
                    "tensors": {key: {"dtype": dtype, "shape": shape, "sha256": digest(raw)}
                                for key, (dtype, shape, raw) in output_rows.items()}}
        target = self.root / "real-output.safetensors"
        model._write_model(target, model.p.scan_safetensors(source_file), identity)
        self.assertEqual(target.read_bytes(), output_bytes)
        identity["tensors"][key]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "quantized tensor differs"):
            model._write_model(self.root / "bad-output.safetensors", model.p.scan_safetensors(source_file), identity)


if __name__ == "__main__":
    unittest.main()
