"""CPU reconstruction tests; tiny identities stand in for model assets."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_mxfp8_prepare_model as fixtures

prepare, write_safetensors, sha = fixtures.prepare, fixtures.write_safetensors, fixtures.sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_mach.mxfp8 import reconstruct_model as reconstruct


class ReconstructionCPU(fixtures.MaterializerCPU):
    def setUp(self):
        super().setUp()
        patcher = patch.object(reconstruct, "p", prepare)
        patcher.start(); self.addCleanup(patcher.stop)
        # Metadata is independently converted, rather than copied from MXFP8.
        source_hashes = {"config.json": prepare.file_sha256(self.bf16 / "config.json")}
        generated = {name: (self.mx8 / name).read_text()
                     for name in ("tokenizer_config.json", "generation_config.json")}
        for name in ("tokenizer.json", "chat_template.jinja"):
            (self.bf16 / name).write_bytes((self.mx8 / name).read_bytes())
        for name, value in (("BF16_METADATA_HASHES", source_hashes),
                            ("MXFP8_CONFIG_SOURCE", (self.mx8 / "config.json").read_text()),
                            ("GENERATED_ASSETS", generated),
                            ("BF16_TOKENIZER_SHA256", prepare.file_sha256(self.bf16 / "tokenizer.json")),
                            ("BF16_SHARD_IDENTITIES", {prepare.file_sha256(self.bf16 / "model.safetensors"):
                                                       (self.bf16 / "model.safetensors").stat().st_size})):
            patcher = patch.object(reconstruct, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        # Lifecycle tests do not need Torch; arithmetic gets a separate real test.
        _, parent, _ = prepare.scan_checkpoint(self.mx8)
        def model_writer(path, source, replacements):
            values = {}
            for key, row in prepare._expected_output_tensors().items():
                if key in replacements:
                    item = replacements[key]
                elif key in prepare.selected_keys():
                    item = source[prepare.resolve_bf16_key(key, source)]
                else:
                    item = parent[key]
                begin, end = item["data_offsets"]
                raw = b"".join(prepare._regions(item["path"], item["base"] + begin, end - begin))
                values[key] = (item["dtype"], item["shape"], raw)
            write_safetensors(path, values)
        self.model_writer = patch.object(reconstruct, "_write_model", side_effect=model_writer)
        self.model_writer.start(); self.addCleanup(self.model_writer.stop)
        def tokenizer_writer(bf16, stage):
            for name in ("tokenizer.json", "chat_template.jinja"):
                (stage / name).write_bytes((bf16 / name).read_bytes())
            return {"method": "mock", "transformers": "test", "tokenizers": "test"}
        patcher = patch.object(reconstruct, "_prepare_tokenizer", side_effect=tokenizer_writer)
        self.real_tokenizer_writer = patcher.get_original()[0]
        patcher.start(); self.addCleanup(patcher.stop)

    def build(self):
        return reconstruct.reconstruct_model(self.bf16, self.codes, self.output)

    def test_quantization_semantic_drift_rejected_before_materialization(self):
        original = json.loads(reconstruct.MXFP8_CONFIG_SOURCE)
        for field, value in (("targets", ["Conv2d"]), ("weights.dynamic", True),
                             ("weights.symmetric", False), ("input_activations.dynamic", False),
                             ("input_activations.symmetric", False)):
            with self.subTest(field=field):
                config = copy.deepcopy(original)
                owner = config["quantization_config"]["config_groups"]["group_0"]
                parts = field.split(".")
                for name in parts[:-1]:
                    owner = owner[name]
                owner[parts[-1]] = value
                with patch.object(reconstruct, "MXFP8_CONFIG_SOURCE", json.dumps(config)):
                    with self.assertRaisesRegex(ValueError, "quantization configuration differs"):
                        self.build()
                self.assertFalse(self.output.exists())

    def test_full_materialization_preserves_scales_and_copies_assets(self):
        manifest = self.build()
        self.assertTrue(prepare.validate_model(self.output, verify_weights=True)["verified_weights"])
        self.assertNotIn("mxfp8", manifest["inputs"])
        self.assertTrue(manifest["original_mxfp8_tensor_bytes_verified"])
        self.assertNotIn(self.temp.name, str(manifest))
        self.assertEqual((self.output / "tokenizer_config.json").read_bytes(),
                         (self.mx8 / "tokenizer_config.json").read_bytes())
        self.assertFalse((self.output / "model.safetensors.index.json").exists())

    def test_failure_cleans_only_the_new_stage(self):
        unrelated = self.output.parent / ".output.building-unrelated"
        unrelated.mkdir()
        with patch.object(reconstruct, "_write_model", side_effect=ValueError("reconstruction hash differs")), \
             self.assertRaisesRegex(ValueError, "hash differs"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".output.building-*")), [unrelated])

    def test_metadata_mismatch_is_rejected_before_writing(self):
        (self.bf16 / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "fixed release asset"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(".output.building-*")))

    def test_standard_tokenizer_conversion_hash_gate_and_temporary_cleanup(self):
        calls = []
        class FakeTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((path, kwargs))
                return FakeTokenizer()
            def save_pretrained(self, path):
                target = Path(path)
                for name in ("tokenizer.json", "chat_template.jinja"):
                    (target / name).write_bytes((self_source / name).read_bytes())
                config = json.loads(reconstruct.GENERATED_ASSETS["tokenizer_config.json"])
                config["local_files_only"] = True
                (target / "tokenizer_config.json").write_text(json.dumps(config))
                (target / "unrelated.json").write_text("excluded")
        self_source = self.bf16
        self.output.mkdir()
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=FakeTokenizer)}), \
             patch.object(reconstruct, "version", return_value="test-version"):
            result = self.real_tokenizer_writer(self.bf16, self.output)
            self.assertEqual(result["transformers"], "test-version")
            self.assertEqual(calls, [(str(self.bf16), {"local_files_only": True})])
            self.assertEqual({path.name for path in self.output.iterdir()}, {"tokenizer.json", "chat_template.jinja"})
            with patch.object(prepare, "HF_ASSET_HASHES", {"tokenizer.json": "0" * 64}), \
                 self.assertRaisesRegex(ValueError, "conversion hash differs"):
                self.real_tokenizer_writer(self.bf16, self.output)
            self.assertFalse(list(self.output.glob(".tokenizer-*")))


class ArithmeticCPU(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
    def test_bf16_observer_and_norm_arithmetic(self):
        import torch
        # Boundary includes BF16 mantissa carry at 1.75, zero, and negatives.
        maxima = [0., 1., 1.7421875, 1.75, 2., 3.5, 448.]
        x = torch.tensor([[value, -value] * 16 for value in maxima], dtype=torch.bfloat16)
        codes, scales = reconstruct.quantize_block(x)
        rounded = ((x.reshape(-1, 1, 32).abs().amax(-1).view(torch.uint16).int()
                    + 32) & 0xff80).to(torch.uint16).view(torch.bfloat16)
        expected_scales = (127 + torch.floor(torch.log2(rounded)) - 8).byte()
        expected_codes = (x.float() / torch.exp2(expected_scales.float() - 127)
                          .repeat_interleave(32, -1)).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(scales, expected_scales))
        self.assertTrue(torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8)))
        self.assertEqual(scales.flatten().tolist(), [0, 119, 119, 120, 120, 121, 128])
        y = torch.tensor([.001, .002, -.003, .3], dtype=torch.float32)
        plus = reconstruct.convert_norm("model.language_model.norm.weight", y, torch.bfloat16)
        self.assertTrue(torch.equal(plus, ((y.float() + 1).bfloat16().float() - 1).bfloat16()))
        linear = reconstruct.convert_norm("model.language_model.layers.0.linear_attn.norm.weight", y, torch.bfloat16)
        self.assertTrue(torch.equal(linear, y.bfloat16()))
        a_log = reconstruct.convert_norm("model.language_model.layers.0.linear_attn.A_log", y, torch.bfloat16)
        self.assertTrue(torch.equal(a_log, y.bfloat16()))
        with self.assertRaises(ValueError):
            reconstruct.quantize_block(x.float())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
    def test_real_streamed_writer_checks_original_and_final_hashes(self):
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "model.language_model.layers.0.mlp.gate_proj.weight"
            norm = "model.language_model.norm.weight"
            a_log = "model.language_model.layers.0.linear_attn.A_log"
            x = torch.arange(96, dtype=torch.float32).reshape(3, 32).div(31).bfloat16()
            codes, scales = reconstruct.quantize_block(x)
            replacement = torch.zeros_like(codes)
            y = torch.tensor([.002, -.003], dtype=torch.bfloat16)
            z = torch.tensor([1., 2.], dtype=torch.float32)
            raw_x, raw_y, raw_z = reconstruct._raw(x), reconstruct._raw(y), reconstruct._raw(z)
            write_safetensors(root / "source.safetensors", {
                key: ("BF16", [3, 32], raw_x), norm: ("BF16", [2], raw_y), a_log: ("F32", [2], raw_z)})
            write_safetensors(root / "l0.safetensors", {key: ("F8_E4M3", [3, 32], reconstruct._raw(replacement))})
            source, l0 = prepare.scan_safetensors(root / "source.safetensors"), prepare.scan_safetensors(root / "l0.safetensors")
            quant = {key: {"shape": [3, 32], "source_sha256": sha(raw_x),
                          "stored_values_sha256": sha(reconstruct._raw(codes)),
                          "stored_scales_sha256": sha(reconstruct._raw(scales))}}
            retained = {k: {"shape": [2], "dtype": "BF16", "source_key": k, "sha256": sha(reconstruct._raw(
                        reconstruct.convert_norm(k, v, torch.bfloat16)))} for k, v in ((norm, y), (a_log, z))}
            with patch.object(reconstruct, "p", prepare), patch.object(prepare, "QUANTIZED", quant), \
                 patch.object(prepare, "RETAINED", retained), patch.object(prepare, "L0_REPLACEMENTS", {key: sha(reconstruct._raw(replacement))}), \
                 patch.object(prepare, "selected_keys", return_value=set()), patch.object(reconstruct, "CHUNK_BYTES", 64):
                reconstruct._write_model(root / "output.safetensors", source, l0)
                actual = prepare.scan_safetensors(root / "output.safetensors")
                self.assertEqual(prepare.tensor_sha256(actual[prepare.scale_key(key)]), quant[key]["stored_scales_sha256"])
                self.assertEqual(prepare.tensor_sha256(actual[key]), sha(reconstruct._raw(replacement)))
                self.assertEqual(actual[a_log]["dtype"], "BF16")
                quant[key]["stored_values_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "Original MXFP8 reconstruction hash differs"):
                    reconstruct._write_model(root / "bad.safetensors", source, l0)


if __name__ == "__main__":
    unittest.main()
