"""CPU-only staging, hash, exact-patch and rollback contracts."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/install.py"


def load():
    spec = importlib.util.spec_from_file_location("mach_mxfp8_installer_cpu", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallContracts(unittest.TestCase):
    def setUp(self):
        self.installer = load()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "site"
        self.data = self.root / "profile"
        self.data.mkdir()
        self.before = {"vllm/one.py": b"value = 1\n", "vllm/two.py": b"value = 2\n",
                       "vllm/guard.py": b"unchanged = True\n"}
        self.after = {**self.before, "vllm/one.py": b"value = 10\n", "vllm/two.py": b"value = 20\n"}
        text = "".join("".join(difflib.unified_diff(self.before[name].decode().splitlines(True),
                self.after[name].decode().splitlines(True), fromfile="a/" + name, tofile="b/" + name))
                for name in self.before if self.before[name] != self.after[name])
        manifest = {"schema_version": 1, "profile": self.installer.PROFILE, "packages": {},
                    "patch_sha256": self.installer.sha256(text.encode()), "files": {
            name: {"upstream_sha256": self.installer.sha256(self.before[name]),
                   "installed_sha256": self.installer.sha256(self.after[name])}
            for name in self.before}}
        (self.data / "runtime.patch").write_text(text)
        (self.data / "runtime_sources.json").write_text(json.dumps(manifest))
        for name, value in self.before.items():
            target = self.site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            target.chmod(0o640)

    def run_install(self, **kwargs):
        return self.installer.install_profile(self.site, verify_packages=False,
                                              data_directory=self.data, **kwargs)

    def test_dry_run_and_exact_repeat_install(self):
        result = self.run_install()
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["state"], "upstream")
        self.assertEqual(len(result["changes"]), 2)
        self.assertFalse((self.site / ".vllm-mach-mxfp8-install.lock").exists())
        for name, value in self.before.items():
            self.assertEqual((self.site / name).read_bytes(), value)
        self.assertTrue(self.run_install(apply=True)["applied"])
        for name, value in self.after.items():
            self.assertEqual((self.site / name).read_bytes(), value)
            self.assertEqual((self.site / name).stat().st_mode & 0o777, 0o640)
        result = self.run_install(apply=True)
        self.assertEqual(result["state"], "installed")
        self.assertEqual(result["changes"], [])
        self.assertFalse(result["applied"])

    def test_reject_drift_mixed_states_and_guard_drift(self):
        for name, value in [("vllm/one.py", b"drift = True\n"),
                            ("vllm/one.py", self.after["vllm/one.py"]),
                            ("vllm/guard.py", b"drift = True\n")]:
            (self.site / name).write_bytes(value)
            with self.assertRaises(RuntimeError):
                self.run_install(apply=True)
            (self.site / name).write_bytes(self.before[name])
        self.assertEqual((self.site / "vllm/two.py").read_bytes(), self.before["vllm/two.py"])

    def test_rollback_on_second_publication_failure(self):
        original = self.installer._replace
        calls = []
        def fail_second(target, value, mode):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("injected publication failure")
            original(target, value, mode)
        with patch.object(self.installer, "_replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                self.run_install(apply=True)
        for name, value in self.before.items():
            self.assertEqual((self.site / name).read_bytes(), value)
        self.assertFalse((self.site / ".vllm-mach-mxfp8-install.lock").exists())
        self.assertFalse(list(self.site.rglob(".mach-mxfp8-*")))

    def test_exact_hunks_reject_fuzz_offsets_and_path_escape(self):
        text = (self.data / "runtime.patch").read_bytes()
        changed_context = {**self.before, "vllm/one.py": b"# offset\nvalue = 1\n"}
        with self.assertRaisesRegex(RuntimeError, "context mismatch"):
            self.installer.apply_patch(changed_context, text)
        with self.assertRaises(RuntimeError):
            self.installer.apply_patch(self.before, text.replace(b"a/vllm/one.py", b"a/vllm/../one.py"))
        self.assertEqual(self.installer.apply_patch(self.before, text), self.after)

    def test_artifact_checksum_and_package_versions_fail_closed(self):
        (self.data / "runtime.patch").write_bytes(b"corrupt patch\n")
        with self.assertRaisesRegex(RuntimeError, "patch SHA256"):
            self.run_install(apply=True)
        with patch.object(self.installer.metadata, "version", return_value="0.30.0"):
            with self.assertRaisesRegex(RuntimeError, "requires vllm==0.29.0"):
                self.installer.check_packages({"vllm": "0.29.0"})
        with patch.object(self.installer.metadata, "version", return_value="2.13.0+cu130"):
            self.assertEqual(self.installer.check_packages({"torch": "2.13.0"}),
                             {"torch": "2.13.0+cu130"})

    def test_lock_and_symlink_fail_before_source_publication(self):
        lock = self.site / ".vllm-mach-mxfp8-install.lock"
        lock.write_text("other installer")
        with self.assertRaisesRegex(RuntimeError, "already locked"):
            self.run_install(apply=True)
        lock.unlink()
        target = self.site / "vllm/guard.py"
        target.unlink()
        target.symlink_to(self.site / "vllm/one.py")
        with self.assertRaisesRegex(RuntimeError, "symlinked"):
            self.run_install(apply=True)

    def test_bundled_profile_is_self_contained_and_has_frozen_compute_guards(self):
        manifest = self.installer.load_manifest()
        self.assertEqual(manifest["packages"]["vllm"], "0.29.0")
        self.assertNotIn("mxfp6-sm120", manifest["packages"])
        for name, entry in manifest["files"].items():
            self.assertTrue(name.startswith("vllm/"))
            self.assertEqual(len(entry["installed_sha256"]), 64)
        gdn = manifest["files"]["vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"]
        self.assertEqual(gdn["installed_sha256"], gdn["frozen_champion_sha256"])
        fla = manifest["files"]["vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"]
        self.assertEqual(fla["upstream_sha256"], fla["installed_sha256"])
        patch_text = (self.installer.DATA / "runtime.patch").read_text()
        for forbidden in ("/task", "vllm.mach.", "head_helpers", "experimental_overlay"):
            self.assertNotIn(forbidden, patch_text)


if __name__ == "__main__":
    unittest.main()
