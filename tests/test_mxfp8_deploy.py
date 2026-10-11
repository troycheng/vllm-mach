"""CPU-only source admission and CUDA header-precedence build contracts."""
import importlib.util
import ast
import json
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def load_builder(filename="build-mxfp8.py"):
    sys.path.insert(0, str(DEPLOY))
    try:
        spec = importlib.util.spec_from_file_location("mach_build_cpu", DEPLOY / filename)
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
            with self.assertRaisesRegex(RuntimeError, "Expected one CUDA 13"):
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
        self.assertEqual([name[:4] for name in builder.PATCHES], ["0001", "0003", "0004", "0005", "0006"])
        docker = (DEPLOY / "Dockerfile.mxfp8").read_text()
        self.assertNotIn("apt-get", docker)
        self.assertNotIn("native/ar_norm", docker)
        self.assertIn('ENTRYPOINT ["vllm-mach-mxfp8-serve"]', docker)
        self.assertIn("--no-deps --no-build-isolation", docker)
        self.assertIn("python3 -m vllm_mach.mxfp8.install --apply", docker)
        self.assertIn("ARG MXFP6_REVISION=" + load_builder("build-mxfp6.py").REVISION, docker)

    def test_both_builders_accept_soname_only_and_deduplicate_aliases(self):
        for filename in ("build-mxfp6.py", "build-mxfp8.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as td:
                builder = load_builder(filename)
                root = Path(td)
                cuda = root / "cuda"
                (cuda / "bin").mkdir(parents=True)
                (cuda / "bin/nvcc").write_text("compiler")
                library = cuda / "lib64/libnvrtc.so.13"
                library.parent.mkdir()
                library.write_text("library")
                site = root / "site"
                include = site / "nvidia/cu13/include"
                include.mkdir(parents=True)
                for name in ("cuda_fp16.h", "cuda_bf16.h"):
                    (include / name).write_text("header")
                first = builder.discover_toolchain(cuda, [site])
                library.with_name("libnvrtc.so").symlink_to(library.name)
                self.assertEqual(builder.discover_toolchain(cuda, [site]), first)
                args = builder.cmake_arguments(root / "cutlass", first)
                self.assertIn("-DCUDA_NVRTC_LIB=" + str(library.resolve()), args)
                self.assertEqual(shlex.split(shlex.join(args)), args)
                self.assertEqual([name[:4] for name in builder.PATCHES], ["0001", "0003", "0004", "0005", "0006"])

    def test_no_gpu_smoke_requires_loaded_ba_and_bool_pdl_schema(self):
        builder = load_builder("build-mxfp6.py")
        tree = ast.parse((DEPLOY / "build-mxfp8.py").read_text())
        probe = next(node.value.value for node in ast.walk(tree)
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "probe" for target in node.targets))
        with tempfile.TemporaryDirectory() as td:
            library = Path(td) / "native.so"
            library.write_bytes(b"cpu-test")
            mx6 = types.ModuleType("mxfp6")
            loads = []
            mx6.load_library = lambda: loads.append("mx6")
            mx6.gemm_from_gdn = lambda: None
            mx6.mxfp8 = types.SimpleNamespace(load_library=lambda: (loads.append("mx8") or library))
            mx6.mxfp8_dual = types.SimpleNamespace(load_library=lambda: library,
                                                 SUPPORTED_SHAPES=(), select_variant=lambda *args: True)
            schema = lambda text: types.SimpleNamespace(default=types.SimpleNamespace(_schema=text))
            ops = types.SimpleNamespace(
                bf16_ba_m16=schema("bf16_ba_m16(Tensor x, Tensor weight) -> Tensor"),
                gemm=schema("gemm(Tensor a, bool use_pdl=False) -> Tensor"),
                allocate_workspace=schema("workspace"), dual_gemm_out=schema("dual"))
            torch = types.ModuleType("torch")
            torch.ops = types.SimpleNamespace(mxfp8_sm120=ops,
                mxfp6=types.SimpleNamespace(gemm_from_swiglu=object(), gemm_w6a8_pdl=object()))
            with patch.dict(sys.modules, {"torch": torch, "mxfp6": mx6}):
                for smoke in (builder.SMOKE, probe):
                    exec(smoke, {})
                    self.assertIn("mx8", loads)
                    ops.gemm.default._schema = "gemm(Tensor a) -> Tensor"
                    with self.assertRaises(AssertionError):
                        exec(smoke, {})
                    ops.gemm.default._schema = "gemm(Tensor a, bool use_pdl=False) -> Tensor"
                    with patch.object(ops, "bf16_ba_m16", schema("wrong(Tensor a) -> Tensor")):
                        with self.assertRaises(AssertionError):
                            exec(smoke, {})


if __name__ == "__main__":
    unittest.main()
