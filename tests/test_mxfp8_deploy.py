"""CPU-only source admission and CUDA header-precedence build contracts."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def load_builder():
    sys.path.insert(0, str(DEPLOY))
    try:
        spec = importlib.util.spec_from_file_location("mach_mxfp8_build_cpu", DEPLOY / "build-mxfp8.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class BuildContracts(unittest.TestCase):
    def test_nvrtc_and_system_include_discovery(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cuda = root / "cuda-13.0"
            nvcc = cuda / "bin/nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("compiler")
            library = cuda / "targets/x86_64-linux/lib/libnvrtc.so.13.0.88"
            library.parent.mkdir(parents=True)
            library.write_text("library")
            site = root / "dist-packages"
            include = site / "nvidia/cu13/include"
            include.mkdir(parents=True)
            for name in ("cuda_fp16.h", "cuda_bf16.h"):
                (include / name).write_text("header")
            found = builder.discover_toolchain(cuda, [site])
            self.assertEqual(found["nvrtc"], str(library.resolve()))
            arguments = builder.cmake_arguments(root / "cutlass", found)
            flags = next(arg for arg in arguments if arg.startswith("-DCMAKE_CUDA_FLAGS="))
            self.assertEqual(flags, "-DCMAKE_CUDA_FLAGS=-isystem " + str(include.resolve()))
            self.assertNotIn("-I" + str(include), flags)
            self.assertEqual(shlex.split(shlex.join(arguments)), arguments)
            library.write_text("first")
            library.with_name("libnvrtc.so.13.0.99").write_text("second")
            with self.assertRaisesRegex(RuntimeError, "Expected one CUDA 13.0"):
                builder.discover_toolchain(cuda, [site])

    def test_clean_snapshot_does_not_copy_old_binaries_or_cache(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            (source / "input.cu").write_text("source")
            for name in ("build/old.cu", "python/old.so", "python/__pycache__/old.pyc",
                         "third_party/cutlass/old.h", "package.egg-info/old.txt", "CMakeCache.txt"):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old")
            receipt = builder.copy_source(source, root / "clean")
            self.assertEqual(set(receipt["source_sha256"]), {"input.cu"})
            self.assertEqual(set(builder.source_hashes(root / "clean")), {"input.cu"})

    def test_old_revision_and_missing_source_are_rejected_without_download(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                builder.validate_source(Path(td))
            with patch.object(builder.urllib.request, "urlretrieve") as download:
                with self.assertRaisesRegex(RuntimeError, "full 40-character"):
                    builder.archive(builder.REPOSITORY, "main", Path(td) / "archive")
                download.assert_not_called()
        self.assertEqual(builder.CUTLASS, "e6233cbac5d7c7a865c19c91cd684ceece19513c")
        self.assertEqual([name[:4] for name in builder.PATCHES], ["0001", "0003", "0004", "0005"])
        docker = (DEPLOY / "Dockerfile.mxfp8").read_text()
        self.assertNotIn("apt-get", docker)
        self.assertNotIn("native/ar_norm", docker)
        self.assertIn('ENTRYPOINT ["vllm-mach-mxfp8-serve"]', docker)
        self.assertIn("--no-deps --no-build-isolation", docker)
        self.assertIn("python3 -m vllm_mach.mxfp8.install --apply", docker)


if __name__ == "__main__":
    unittest.main()
