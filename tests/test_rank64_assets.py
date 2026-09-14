"""Manifest checks run without vLLM, PyTorch or a GPU."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_SOURCE = Path(__file__).resolve().parents[1] / "src/vllm_mach/exl3/rank64_assets.py"
_SPEC = importlib.util.spec_from_file_location("rank64_assets_test", _SOURCE)
assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assets)


def manifest():
    return {
        "schema": assets.SCHEMA, "mask": list(assets.MASK),
        "geometry": {"rows": 32, "hidden": 5120, "gate_up": 17408, "rank": 64, "tp": 2},
        "official_config_sha256": assets.OFFICIAL_CONFIG,
        "official_index_sha256": assets.OFFICIAL_INDEX,
        "entries": [{"layer": layer, "rank": rank, "activation_global_scale": 32.,
                     "official_bf16_gu_sha256": "a" * 64, "norm_bf16_sha256": "b" * 64}
                    for layer in assets.MASK for rank in (0, 1)],
    }


class Rank64ManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scales = {"config": {"hidden_size": 5120, "num_hidden_layers": 64, "rms_norm_eps": 1e-6},
                  "layers": {str(layer): {"raw_payload_sha256": "b" * 64,
                  "selected_power_of_two_global_scale": 32.} for layer in assets.MASK}}
        path = self.root / "static_scales.json"
        path.write_text(json.dumps(scales))
        patched = patch.object(assets, "SCALE_RECEIPT", assets.sha256(path))
        patched.start()
        self.addCleanup(patched.stop)
        content_patch = patch.object(assets, "SCALE_CONTENT", assets.static_scale_content(scales))
        content_patch.start()
        self.addCleanup(content_patch.stop)

    def check(self, value):
        (self.root / "manifest.json").write_text(json.dumps(value))
        return assets.load_manifest(self.root)

    def test_exact_mask_and_tp2(self):
        data = self.check(manifest())
        self.assertEqual(len(data["entries"]), 96)

    def test_regenerated_report_metadata_is_allowed(self):
        path = self.root / "static_scales.json"
        report = json.loads(path.read_text())
        report["source"] = {"config": "/another/model/config.json", "script": "/another/tool.py"}
        path.write_text(json.dumps(report, indent=4))
        data = manifest()
        data["static_scales_sha256"] = assets.sha256(path)
        self.assertEqual(len(self.check(data)["entries"]), 96)

    def test_regenerated_report_cannot_change_numerical_content(self):
        path = self.root / "static_scales.json"
        report = json.loads(path.read_text())
        report["layers"]["0"]["selected_power_of_two_global_scale"] = 16.
        path.write_text(json.dumps(report))
        data = manifest()
        data["static_scales_sha256"] = assets.sha256(path)
        data["entries"][0]["activation_global_scale"] = 16.
        with self.assertRaisesRegex(ValueError, "numerical content"):
            self.check(data)

    def test_regenerated_report_still_checks_file_hash(self):
        data = manifest()
        data["static_scales_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.check(data)

    def test_relative_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            assets.load_manifest(Path("relative"))

    def test_missing_manifest_explains_import(self):
        with self.assertRaisesRegex(ValueError, "import_rank64_bundle"):
            assets.load_manifest(self.root)

    def test_incomplete_or_duplicate_entries(self):
        for mutation in (lambda d: d["entries"].pop(),
                         lambda d: d["entries"].__setitem__(1, copy.deepcopy(d["entries"][0]))):
            data = manifest()
            mutation(data)
            with self.assertRaisesRegex(ValueError, "exactly"):
                self.check(data)

    def test_other_mask_and_geometry(self):
        for field, value in (("mask", list(range(48))), ("geometry", {"rows": 24})):
            data = manifest()
            data[field] = value
            with self.assertRaises(ValueError):
                self.check(data)

    def test_checkpoint_identity(self):
        data = manifest()
        data["official_config_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.check(data)

    def test_scale_not_retuned(self):
        data = manifest()
        data["entries"][0]["activation_global_scale"] = 64
        with self.assertRaisesRegex(ValueError, "analytic"):
            self.check(data)

    def test_boolean_rank_is_not_integer(self):
        data = manifest()
        data["entries"][0]["rank"] = False
        with self.assertRaises(ValueError):
            self.check(data)

    def test_norm_identity_required(self):
        data = manifest()
        data["entries"][0].pop("norm_bf16_sha256")
        with self.assertRaisesRegex(ValueError, "norm"):
                self.check(data)

    def test_scale_receipt_binds_layer(self):
        data = manifest()
        data["entries"][0]["activation_global_scale"] = 16
        with self.assertRaisesRegex(ValueError, "static scale receipt"):
            self.check(data)

    def test_asset_integrity(self):
        path = self.root / "asset"
        path.write_bytes(b"weights")
        digest = assets.sha256(path)
        self.assertEqual(assets.checked_path(self.root, "asset", digest), path.resolve())
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            assets.checked_path(self.root, "asset", digest)

    def test_asset_escape_and_symlink(self):
        (self.root / "link").symlink_to(self.root.parent)
        for name in ("../asset", "/asset", "link/asset"):
            with self.assertRaises(ValueError):
                assets.checked_path(self.root, name, "a" * 64)


if __name__ == "__main__":
    unittest.main()
