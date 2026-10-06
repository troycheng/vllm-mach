"""CPU-only contracts for the native opt-in, without torch/vLLM installed."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1] / "src" / "vllm_mach"
PREFIXES = ("vllm", "torch", "mxfp6")


def load(name: str, filename: str):
    path = HERE / filename if filename == "plugin.py" else HERE / "mxfp8" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class PackageGlueTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: module for name, module in sys.modules.items()
                      if name.startswith(PREFIXES)}
        for name in self.saved:
            sys.modules.pop(name, None)
        package = types.ModuleType("vllm_mach")
        package.__path__ = [str(HERE)]
        sys.modules["vllm_mach"] = package
        mxfp8 = types.ModuleType("vllm_mach.mxfp8")
        mxfp8.__path__ = [str(HERE / "mxfp8")]
        sys.modules[mxfp8.__name__] = mxfp8
        mxfp6 = types.ModuleType("vllm_mach.mxfp6")
        self.mx6_calls = []
        mxfp6.register_dense_kernel = lambda: self.mx6_calls.append("mx6")
        sys.modules[mxfp6.__name__] = mxfp6

    def tearDown(self):
        for name in list(sys.modules):
            if name.startswith(PREFIXES):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved)

    def test_opt_out_does_not_import_worker_or_cuda_backend(self):
        plugin = load("vllm_mach.plugin", "plugin.py")
        plugin.version = lambda name: "0.29.0"
        with patch.dict(os.environ, {"VLLM_MACH_NATIVE_MXFP8": "0"}):
            plugin.register()
        self.assertEqual(self.mx6_calls, ["mx6"])
        self.assertNotIn("vllm_mach.mxfp8.worker", sys.modules)
        self.assertNotIn("vllm_mach.mxfp8.native_backend", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_package_entrypoint_import_does_not_import_torch(self):
        package = load("vllm_mach.mxfp8", "__init__.py")
        self.assertTrue(callable(package.install_worker_backend))
        self.assertTrue(callable(package.verify_worker_execution))
        self.assertNotIn("vllm_mach.mxfp8.native_backend", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_repeat_registration_wraps_once_and_original_init_runs_first(self):
        events = []
        class Worker:
            def init_device(self):
                events.append("original_init")
                return "ready"
            def compile_or_warm_up_model(self):
                events.append("original_compile")
                return "compiled"
        gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker.Worker = Worker
        sys.modules[gpu_worker.__name__] = gpu_worker
        backend = types.ModuleType("vllm_mach.mxfp8.native_backend")
        backend.verify_runtime = lambda: events.append("verify_binary_schema")
        backend.install = lambda: events.append("install_backend")
        backend.inspect = lambda worker: {"installed": True,
                                          "counts": {"native_calls": 1}}
        sys.modules[backend.__name__] = backend
        plugin = load("vllm_mach.plugin", "plugin.py")
        plugin.version = lambda name: "0.29.0"
        with patch.dict(os.environ, {"VLLM_MACH_NATIVE_MXFP8": "1"}):
            plugin.register()
            plugin.register()
        self.assertEqual(self.mx6_calls, ["mx6", "mx6"])
        self.assertEqual(Worker().init_device(), "ready")
        self.assertEqual(events, ["original_init", "verify_binary_schema",
                                  "install_backend"])
        self.assertEqual(Worker().compile_or_warm_up_model(), "compiled")

    def test_original_init_failure_never_loads_backend(self):
        events = []
        class Worker:
            def init_device(self):
                events.append("original_init")
                raise RuntimeError("device failed")
            def compile_or_warm_up_model(self):
                return None
        gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker.Worker = Worker
        sys.modules[gpu_worker.__name__] = gpu_worker
        backend = types.ModuleType("vllm_mach.mxfp8.native_backend")
        backend.verify_runtime = lambda: events.append("verify_binary_schema")
        backend.install = lambda: events.append("install_backend")
        sys.modules[backend.__name__] = backend
        worker_module = load("vllm_mach.mxfp8.worker", "worker.py")
        worker_module.install_worker_hook()
        with self.assertRaisesRegex(RuntimeError, "device failed"):
            Worker().init_device()
        self.assertEqual(events, ["original_init"])

    def test_missing_or_bad_schema_fails_without_experiment_receipt(self):
        torch = types.ModuleType("torch")
        loaded = []
        torch.ops = types.SimpleNamespace(load_library=lambda path: loaded.append(path))
        sys.modules["torch"] = torch
        backend = load("vllm_mach.mxfp8.native_backend", "native_backend.py")
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "mxfp8_torch.so"
            with patch.dict(os.environ, {"MXFP8_LIBRARY_PATH": str(library)}):
                with self.assertRaisesRegex(ImportError, "does not exist"):
                    backend.verify_runtime()
                library.write_bytes(b"test-only-binary")
                with self.assertRaisesRegex(RuntimeError, "operator missing: gemm"):
                    backend.verify_runtime()
                torch.ops.mxfp8_sm120 = types.SimpleNamespace(gemm=types.SimpleNamespace(
                    default=types.SimpleNamespace(_schema="mxfp8_sm120::gemm(Tensor a) -> Tensor")))
                with self.assertRaisesRegex(RuntimeError, "schema mismatch for gemm"):
                    backend.verify_runtime()
                torch.ops.mxfp8_sm120.gemm.default._schema = (
                    "mxfp8_sm120::gemm(Tensor a, Tensor b, Tensor sa, Tensor sb, "
                    "Tensor(a!)? out=None, Tensor? workspace=None, int tactic=-1, "
                    "int splits=1, int swizzle=1, int sms=0) -> Tensor(a!)")
                torch.ops.mxfp8_sm120.allocate_workspace = types.SimpleNamespace(
                    default=types.SimpleNamespace(_schema=(
                        "mxfp8_sm120::allocate_workspace(Tensor a, Tensor b, Tensor sa, "
                        "Tensor sb, Tensor out, int tactic, int splits=1, int swizzle=1, "
                        "int sms=0) -> Tensor?")))
                backend.verify_runtime()
                self.assertTrue(backend._NATIVE_LOADED)
                self.assertEqual(backend.inspect()["binary"]["sha256"],
                                 hashlib.sha256(library.read_bytes()).hexdigest())
                self.assertIsNone(backend.inspect()["binary"]["metadata"])
                with self.assertRaisesRegex(RuntimeError, "operator missing: dual_gemm_out"):
                    backend.verify_runtime(require_dual=True)
        self.assertEqual(len(loaded), 3)

    def test_compile_validation_rejects_no_execution_or_capture_miss(self):
        backend = types.ModuleType("vllm_mach.mxfp8.native_backend")
        receipt = {"installed": True, "counts": {}}
        backend.inspect = lambda worker: receipt
        sys.modules[backend.__name__] = backend
        worker = load("vllm_mach.mxfp8.worker", "worker.py")
        with self.assertRaisesRegex(RuntimeError, "no eligible native GEMM"):
            worker.verify_worker_execution()
        receipt["counts"] = {"native_calls": 1, "capture_cache_miss": 1}
        with self.assertRaisesRegex(RuntimeError, "missed warmed workspace"):
            worker.verify_worker_execution()
        receipt["counts"] = {"native_calls": 1}
        self.assertIs(worker.verify_worker_execution(), receipt)
        with self.assertRaisesRegex(RuntimeError, "recorded no native GEMM"):
            worker.verify_worker_execution(require_capture=True)
        receipt["counts"]["native_capture_calls"] = 1
        self.assertIs(worker.verify_worker_execution(require_capture=True), receipt)

    def make_backend(self):
        events = []
        device = types.SimpleNamespace(type="cuda", index=0)

        class Tensor:
            def __init__(self, shape, dtype="bf16", *, contiguous=True, size=1):
                self.shape = shape
                self.dtype = dtype
                self.device = device
                self.ndim = len(shape)
                self.contiguous = contiguous
                self.size = size

            def is_contiguous(self):
                return self.contiguous

            def element_size(self):
                return self.size

            def numel(self):
                result = 1
                for dimension in self.shape:
                    result *= dimension
                return result

            def view(self, shape, *rest):
                shape = shape if isinstance(shape, tuple) else (shape, *rest)
                events.append(("view", shape))
                return Tensor(shape, self.dtype)

            def __add__(self, bias):
                events.append(("bias", bias))
                return self

        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16"
        torch.float8_e4m3fn = "e4m3"
        torch.empty = lambda shape, **kwargs: Tensor(shape, kwargs["dtype"])
        stream = types.SimpleNamespace(cuda_stream=10)
        cuda = types.SimpleNamespace(
            current_device=lambda: 0,
            current_stream=lambda index: stream,
            is_current_stream_capturing=lambda: False,
        )
        torch.cuda = cuda
        workspace = Tensor((7,), size=4)
        allocations = []
        calls = []
        def allocate(*args):
            allocations.append(args)
            return workspace
        torch.ops = types.SimpleNamespace(
            mxfp8_sm120=types.SimpleNamespace(
                allocate_workspace=allocate, gemm=lambda *args: calls.append(args)
            ),
            vllm=types.SimpleNamespace(),
        )
        sys.modules["torch"] = torch
        backend = load("vllm_mach.mxfp8.native_backend", "native_backend.py")
        backend._NATIVE_LOADED = True
        backend._SM120_DEVICES = frozenset({0})
        backend._ORIGINAL_APPLY = lambda *args: "fallback"
        setattr(torch.ops.vllm, backend._OP_NAME, backend._native_impl)
        return backend, torch, Tensor, stream, events, allocations, calls

    def test_import_does_not_load_library_or_query_cuda(self):
        torch = types.ModuleType("torch")
        def forbidden(*args, **kwargs):
            self.fail("import touched CUDA or loaded the library")
        torch.cuda = types.SimpleNamespace(device_count=forbidden,
                                          get_device_capability=forbidden)
        torch.ops = types.SimpleNamespace(load_library=forbidden)
        sys.modules["torch"] = torch
        backend = load("vllm_mach.mxfp8.native_backend", "native_backend.py")
        self.assertFalse(backend._NATIVE_LOADED)
        self.assertFalse(backend._INSTALLED)

    def test_runtime_shape_and_stream_workspaces_have_fresh_outputs(self):
        backend, torch, Tensor, stream, _, allocations, calls = self.make_backend()
        weight, scale = Tensor((256, 128), "e4m3"), Tensor((256, 4), "u8")
        x = Tensor((32, 128), "e4m3")
        first = backend.gemm(x, weight, scale, scale)
        second = backend.gemm(x, weight, scale, scale)
        self.assertIsNot(first, second)
        self.assertIs(allocations[0][4], first)
        self.assertIs(calls[0][4], first)
        self.assertIs(calls[1][4], second)
        self.assertIs(calls[0][5], calls[1][5])
        self.assertEqual(len(allocations), 1)
        self.assertIs(next(iter(backend._WORKSPACES.values())).stream, stream)
        torch.cuda.is_current_stream_capturing = lambda: True
        backend.gemm(x, weight, scale, scale)
        self.assertEqual(backend.inspect()["counts"]["native_capture_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "not warmed"):
            backend.gemm(Tensor((64, 128)), weight, scale, scale)
        self.assertEqual(len(allocations), 1)
        torch.cuda.is_current_stream_capturing = lambda: False
        backend.gemm(Tensor((64, 128)), weight, scale, scale)
        stream.cuda_stream = 11
        backend.gemm(x, weight, scale, scale)
        self.assertEqual(len(allocations), 3)
        self.assertEqual(backend.inspect()["workspace_bytes_total"], 3 * 7 * 4)

    def test_eligible_native_errors_are_not_swallowed_by_fallback(self):
        backend, torch, Tensor, _, events, _, _ = self.make_backend()
        layer = types.SimpleNamespace(
            weight=Tensor((256, 128), "e4m3"), weight_scale=Tensor((256, 4), "u8")
        )
        quant = types.ModuleType(
            "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
        )
        def quantize(x, *, is_sf_swizzled_layout):
            events.append(("quant", is_sf_swizzled_layout))
            return Tensor((32, 128), "e4m3"), Tensor((32, 4), "u8")
        quant.mxfp8_e4m3_quantize = quantize
        sys.modules[quant.__name__] = quant
        bias = object()
        result = backend._apply_weights(None, layer, Tensor((2, 16, 128)), bias)
        self.assertEqual(result.shape, (2, 16, 256))
        self.assertEqual(events, [("view", (-1, 128)), ("quant", True),
                                  ("bias", bias), ("view", (2, 16, 256))])
        def fail(*args):
            raise RuntimeError("native workspace failed")
        torch.ops.mxfp8_sm120.gemm = fail
        with self.assertRaisesRegex(RuntimeError, "native workspace failed"):
            backend._apply_weights(None, layer, Tensor((32, 128)))
        layer.weight.contiguous = False
        self.assertEqual(backend._apply_weights(None, layer, Tensor((32, 128))),
                         "fallback")

    def test_static_guard_fallbacks_do_not_quantize_or_load(self):
        backend, torch, Tensor, _, _, _, _ = self.make_backend()
        def forbidden(*args, **kwargs):
            self.fail("static fallback entered native path")
        backend.gemm = forbidden
        layer = types.SimpleNamespace(
            weight=Tensor((256, 128), "e4m3"), weight_scale=Tensor((256, 4), "u8")
        )
        for x in (Tensor((32, 128), "fp32"), Tensor(()), Tensor((32, 64))):
            self.assertEqual(backend._apply_weights(None, layer, x), "fallback")
        for weight in (Tensor((129, 128), "e4m3"), Tensor((256, 128), "u8"),
                       Tensor((256, 128), "e4m3", contiguous=False)):
            layer.weight = weight
            self.assertEqual(backend._apply_weights(None, layer, Tensor((32, 128))),
                             "fallback")
        layer.weight = Tensor((256, 128), "e4m3")
        for scale in (Tensor((256, 4), "u8", size=2), Tensor((1,), "u8"),
                      Tensor((256, 4), "u8", contiguous=False)):
            layer.weight_scale = scale
            self.assertEqual(backend._apply_weights(None, layer, Tensor((32, 128))),
                             "fallback")

    def test_install_only_patches_selected_cutlass_class_and_is_idempotent(self):
        backend, torch, _, _, _, _, _ = self.make_backend()
        class Cutlass:
            def apply_weights(self, *args):
                return "stock"
        class Other:
            def apply_weights(self, *args):
                return "other"
        original = Cutlass.apply_weights
        module = types.ModuleType("vllm.model_executor.kernels.linear.mxfp8.flashinfer")
        module.FlashInferCutlassMxfp8LinearKernel = Cutlass
        module.OtherKernel = Other
        sys.modules[module.__name__] = module
        utils = types.ModuleType("vllm.utils.torch_utils")
        registrations = []
        utils.direct_register_custom_op = lambda **kwargs: registrations.append(kwargs)
        sys.modules[utils.__name__] = utils
        torch.cuda.device_count = lambda: 2
        torch.cuda.get_device_capability = lambda index: (12, 0) if index == 0 else (9, 0)
        self.assertTrue(backend.install())
        self.assertFalse(backend.install())
        self.assertIs(Cutlass.apply_weights, backend._apply_weights)
        self.assertEqual(Other().apply_weights(), "other")
        self.assertEqual(backend._SM120_DEVICES, frozenset({0}))
        self.assertEqual(len(registrations), 1)
        self.assertEqual(registrations[0]["mutates_args"], [])
        self.assertIs(registrations[0]["fake_impl"], backend._native_fake)
        self.assertTrue(backend.uninstall())
        self.assertIs(Cutlass.apply_weights, original)
        self.assertTrue(backend.install())
        self.assertEqual(len(registrations), 1)

    def test_fake_keeps_runtime_m_symbolic(self):
        backend, _, Tensor, _, _, _, _ = self.make_backend()
        class Symbol:
            def __int__(self):
                raise AssertionError("symbolic M was converted to int")
        m = Symbol()
        output = backend._native_fake(Tensor((m, 128)), Tensor((256, 128)), None, None)
        self.assertIs(output.shape[0], m)
        self.assertEqual(output.shape[1], 256)


if __name__ == "__main__":
    unittest.main()
