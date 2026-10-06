"""CPU-only activation contracts without installing Torch or initializing CUDA."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[1] / "src/vllm_mach/fp8/activation.py")


class Op:
    def __init__(self, callback):
        self.default = self
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)


class Tensor:
    def __init__(self, shape, dtype, *, device=0, strides=None, contiguous=True):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(index=device, type="cuda")
        self.is_cuda = device is not None
        self._strides = strides
        self._contiguous = contiguous

    def stride(self):
        return self._strides

    def is_contiguous(self):
        return self._contiguous


@contextmanager
def fixture():
    events = []
    schemas = []
    fake = []
    original = Op(lambda *args: events.append(("fallback", args)))
    native = Op(lambda *args: events.append(("native", args)))
    namespace = types.SimpleNamespace()
    ops = types.SimpleNamespace(
        _C=types.SimpleNamespace(silu_and_mul_per_block_quant=original),
        vllm_mach_fp8=namespace,
        mach_fp8_activation=types.SimpleNamespace(run=native),
        load_library=lambda path: events.append(("load", path)),
    )
    class Library:
        def __init__(self, name, kind):
            events.append(("library", name, kind))

        def define(self, schema):
            schemas.append(schema)

        def impl(self, name, callback, dispatch_key):
            setattr(namespace, name, Op(callback))

    torch = types.ModuleType("torch")
    torch.ops = ops
    torch.bfloat16, torch.float8_e4m3fn, torch.float32 = "bf16", "e4m3", "f32"
    torch.__version__ = "2.13.0+cu130"
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.library = types.SimpleNamespace(
        Library=Library, register_fake=lambda name, fn: fake.append((name, fn))
    )
    def current_device():
        events.append(("current_device",))
        return 0
    def capability(device):
        events.append(("capability", device))
        return (12, 0)
    torch.cuda = types.SimpleNamespace(
        current_device=current_device, get_device_capability=capability,
        is_current_stream_capturing=lambda: False,
    )
    spec = importlib.util.spec_from_file_location("activation_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.dict(sys.modules, {"torch": torch}), patch.dict(
        os.environ, {"VLLM_MACH_FP8_SILU": "1"}
    ):
        yield module, torch, events, schemas, fake, original


def tensors(rows=32, width=18432, group=128, transposed=True):
    output = Tensor((rows, width // 2), "e4m3")
    input_ = Tensor((rows, width), "bf16")
    scales = Tensor((rows, width // 2 // group), "f32",
                    strides=(1, rows) if transposed else None,
                    contiguous=not transposed)
    return output, input_, scales


class ActivationTests(unittest.TestCase):
    def test_disabled_and_group64_keep_original_op_without_registration(self):
        with fixture() as (module, _, events, _, _, original):
            with patch.dict(os.environ, {"VLLM_MACH_FP8_SILU": "0"}):
                self.assertIs(module.get_fused_op(), original)
                self.assertFalse(module.install())
            self.assertIs(module.get_fused_op(64), original)
            self.assertEqual(events, [])
            self.assertFalse(module.inspect()["registered"])

    def test_schema_registration_is_idempotent_and_does_not_touch_cuda(self):
        with fixture() as (module, _, events, schemas, fake, _):
            op = module.register()
            self.assertIs(module.register(), op)
            self.assertEqual(len(schemas), 1)
            self.assertIn("Tensor(a!) out", schemas[0])
            self.assertIn("Tensor(b!) scales", schemas[0])
            self.assertEqual(len(fake), 1)
            self.assertIsNone(fake[0][1](None, None, None, 128, None, True))
            self.assertEqual(events, [("library", "vllm_mach_fp8", "FRAGMENT")])

    def test_worker_installs_once_and_opaque_op_routes_by_tensor_contract(self):
        with fixture() as (module, torch, events, _, _, _):
            module.verify_runtime = lambda: events.append(("verify",)) or {}
            self.assertTrue(module.install())
            self.assertFalse(module.install())
            op = module.get_fused_op()
            output, input_, scales = tensors()
            op(output, input_, scales, 128, None, True)
            self.assertEqual(events[-1][0], "native")
            self.assertEqual(module.inspect()["counts"],
                             {"native": 1, "eager_native": 1})
            for mutate in (
                lambda: setattr(input_, "dtype", "fp16"),
                lambda: setattr(scales, "_strides", (73, 1)),
                lambda: setattr(output, "device",
                                types.SimpleNamespace(index=1, type="cuda")),
                lambda: setattr(input_, "shape", (32, 1024)),
            ):
                output, input_, scales = tensors()
                mutate()
                op(output, input_, scales, 128, None, True)
                self.assertEqual(events[-1][0], "fallback")
            self.assertEqual(module.inspect()["counts"]["fallback"], 4)
            output, input_, scales = tensors(transposed=False)
            torch.cuda.is_current_stream_capturing = lambda: True
            op(output, input_, scales, 128, None, False)
            self.assertEqual(module.inspect()["counts"]["capture_native"], 1)

    def test_enabled_but_not_installed_or_upper_bound_uses_original(self):
        with fixture() as (module, _, events, _, _, _):
            op = module.get_fused_op()
            output, input_, scales = tensors()
            op(output, input_, scales, 128, None, True)
            self.assertEqual(events[-1][0], "fallback")
            module.verify_runtime = lambda: {}
            module.install()
            op(output, input_, scales, 128, object(), True)
            self.assertEqual(events[-1][0], "fallback")

    def test_missing_wheel_or_incompatible_worker_fails_before_installation(self):
        with fixture() as (module, torch, _, _, _, _):
            with patch.object(module, "version", return_value="0.29.0"), \
                    patch.object(module.importlib.util, "find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "prebuilt"):
                    module.install()
            self.assertFalse(module.inspect()["installed"])
            torch.__version__ = "2.12.0"
            with patch.object(module, "version", return_value="0.29.0"):
                with self.assertRaisesRegex(RuntimeError, "Torch 2.13"):
                    module.verify_runtime()
            torch.__version__ = "2.13.0"
            module.verify_runtime = lambda: {}
            torch.cuda.get_device_capability = lambda device: (9, 0)
            with self.assertRaisesRegex(RuntimeError, "SM120"):
                module.install()
            self.assertFalse(module.inspect()["installed"])

    def test_prebuilt_library_identity_is_verified_and_loaded_once(self):
        with fixture() as (module, torch, events, _, _, _), \
                tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "native.so"
            library.write_bytes(b"test-native")
            torch.ops.mach_fp8_activation.run._schema = (
                "run(Tensor(a!) out, Tensor input, Tensor(b!) scales, "
                "bool is_scale_transposed) -> ()"
            )
            spec = types.SimpleNamespace(origin=str(library))
            with patch.object(module, "version", return_value="0.29.0"), \
                    patch.object(module.importlib.util, "find_spec", return_value=spec):
                identity = module.verify_runtime()
                self.assertEqual(module.verify_runtime(), identity)
            self.assertEqual(identity["path"], str(library.resolve()))
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertEqual(sum(event[0] == "load" for event in events), 1)


if __name__ == "__main__":
    unittest.main()
