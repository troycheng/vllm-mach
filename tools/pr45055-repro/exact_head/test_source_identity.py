"""CPU checks of archive/source integrity without Torch or a fabricated Git HEAD."""

from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from source_identity import HEAD, verify_source


class ArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / ("vllm-" + HEAD)
        self.kernel = self.root / "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
        self.kernel.parent.mkdir(parents=True)
        snapshot = Path(__file__).resolve().parents[1] / "upstream/pr45055_42cff10c.cu"
        shutil.copyfile(snapshot, self.kernel)
        (self.root / "helper.txt").write_text("pristine helper\n")
        (self.root / "alias").symlink_to("helper.txt")
        self.archive = self.base / "official-test.tar.gz"
        self.save_archive()

    def tearDown(self):
        self.temp.cleanup()

    def save_archive(self):
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(self.root, arcname=self.root.name)

    def test_every_file_and_symlink_is_verified_without_claiming_git(self):
        result = verify_source(self.root, self.archive)
        self.assertEqual(result["mode"], "archive")
        self.assertNotIn("verified_git_head", result)
        self.assertEqual(len(result["archive_sha256"]), 64)
        self.assertEqual(result["entries"]["alias"],
                         {"type": "symlink", "target": "helper.txt"})

    def test_modified_helper_is_rejected(self):
        (self.root / "helper.txt").write_text("modified\n")
        with self.assertRaisesRegex(RuntimeError, "Source file differs"):
            verify_source(self.root, self.archive)

    def test_extra_source_file_is_rejected(self):
        (self.root / "extra.cu").write_text("extra\n")
        with self.assertRaisesRegex(RuntimeError, "entries differ"):
            verify_source(self.root, self.archive)

    def test_changed_symlink_is_rejected(self):
        alias = self.root / "alias"
        alias.unlink()
        alias.symlink_to("other.txt")
        with self.assertRaisesRegex(RuntimeError, "Source symlink differs"):
            verify_source(self.root, self.archive)

    def test_wrong_prefix_is_rejected(self):
        different = self.base / "wrong-prefix"
        self.root.rename(different)
        with self.assertRaisesRegex(RuntimeError, "must have prefix"):
            verify_source(different, self.archive)

    def test_matching_tampered_kernel_and_archive_fail_independent_snapshot(self):
        self.kernel.write_text("modified kernel\n")
        self.save_archive()
        with self.assertRaisesRegex(RuntimeError, "independent API snapshot"):
            verify_source(self.root, self.archive)


if __name__ == "__main__":
    unittest.main()
