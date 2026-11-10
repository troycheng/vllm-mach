"""CPU checks for the 4B-only historical norm compilation boundary."""
import ast
import importlib.util
import inspect
import math
import os
from pathlib import Path
import sys
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NormFrontendTests(unittest.TestCase):
    def setUp(self):
        self.frontend = load(ROOT / "src/vllm_mach/mxfp8/norm_frontend.py", "norm_frontend_cpu")
        self.shared = load(ROOT / "tests/fixtures/gdn_output_projection.py", "shared_gdn_cpu")
        self.cls = self.shared.QwenGatedDeltaNetAttention
        self.original = self.cls._output_projection
        self.module_patch = patch.dict(sys.modules, {MODULE: self.shared})
        self.module_patch.start()
        self.env = patch.dict(os.environ, {"VLLM_MACH_PROFILE": self.frontend.PROFILE})
        self.env.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.env.stop)

    def test_preserves_old_fallback_ast_and_real_2d_call_boundary(self):
        # The replacement is exactly the existing non-CUDA fallback plus the
        # common output projection, with no new norm or floating arithmetic.
        original = ast.parse(textwrap.dedent(inspect.getsource(self.original))).body[0]
        expected = ast.Module(body=original.body[1].orelse + original.body[2:], type_ignores=[])
        self.assertEqual(self.frontend._method_body(self.frontend._output_projection),
                         ast.dump(expected, include_attributes=False))
        receipt = self.frontend.install()
        self.assertEqual(receipt["frontend"], "legacy_2d")
        self.assertEqual(len(receipt["body_sha256"]), 64)
        for rows in (1, 16, 64):
            with self.subTest(rows=rows):
                events = []

                class Tensor:
                    def __init__(self, shape, name, contiguous=True):
                        self.shape, self.name, self.contiguous = shape, name, contiguous

                    def reshape(self, *shape):
                        if len(shape) == 1:
                            shape = shape[0]
                        shape = list(shape)
                        if -1 in shape:
                            shape[shape.index(-1)] = math.prod(self.shape) // math.prod(x for x in shape if x != -1)
                        if math.prod(shape) != math.prod(self.shape):
                            raise ValueError("element count changed")
                        events.append((self.name, "reshape", tuple(shape), self.contiguous))
                        return Tensor(tuple(shape), self.name)

                    def flatten(self, start):
                        events.append((self.name, "flatten", start))
                        return Tensor(self.shape[:start] + (math.prod(self.shape[start:]),), self.name)

                instance = self.cls()
                core = Tensor((rows, 32, 128), "core")
                gate = Tensor((rows, 32, 128), "gate", contiguous=False)
                result = object()

                def norm(x, z):
                    events.append(("norm", x.shape, z.shape))
                    self.assertEqual(x.shape, (rows * 32, 128))
                    self.assertEqual(z.shape, x.shape)
                    return x

                def out_proj(x):
                    events.append(("out_proj", x.shape))
                    return result, None

                instance.norm, instance.out_proj = norm, out_proj
                self.assertIs(instance._output_projection(core, gate), result)
                self.assertEqual(events, [
                    ("core", "reshape", (rows * 32, 128), True),
                    ("gate", "reshape", (rows * 32, 128), False),
                    ("norm", (rows * 32, 128), (rows * 32, 128)),
                    ("core", "reshape", (rows, 32, 128), True),
                    ("core", "flatten", -2), ("out_proj", (rows, 4096))])
                self.assertEqual(gate.shape, (rows, 32, 128))
                self.assertFalse(gate.contiguous)

    def test_profile_isolation_and_original_identity_or_ast_drift_rejected(self):
        for profile in ("qwen35-2b-mxfp8-champion-v1", "qwen35-4b-block-fp8-v1", ""):
            with self.subTest(profile=profile), patch.dict(os.environ, {"VLLM_MACH_PROFILE": profile}):
                with self.assertRaisesRegex(RuntimeError, "complete 4B"):
                    self.frontend.install()
                self.assertIs(self.cls._output_projection, self.original)
        source = inspect.getsource(self.original)
        for old, new in (("self.norm(core_attn_out, z)", "self.norm(z, core_attn_out)"),
                         ("flatten(-2)", "flatten(-1)")):
            with self.subTest(mutation=old), patch.object(self.frontend.inspect, "getsource", return_value=source.replace(old, new)):
                with self.assertRaisesRegex(RuntimeError, "AST changed"):
                    self.frontend.install()
                self.assertIs(self.cls._output_projection, self.original)
        self.original.__defaults__ = (None,)
        with self.assertRaisesRegex(RuntimeError, "method identity"):
            self.frontend.install()
        self.original.__defaults__ = None
        self.cls._output_projection = lambda self, core_attn_out, z: core_attn_out
        with self.assertRaisesRegex(RuntimeError, "method identity"):
            self.frontend.install()
        self.assertFalse(self.frontend._INSTALLED)

    def test_repeat_install_is_idempotent_and_replacement_is_detected(self):
        first = self.frontend.install()
        installed = self.cls._output_projection
        self.assertEqual(self.frontend.install(), first)
        self.assertIs(self.cls._output_projection, installed)
        self.assertIs(self.frontend._INSTALLED[self.cls], self.original)
        self.cls._output_projection = self.original
        for operation in (self.frontend.install, self.frontend.inspect_frontend):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(RuntimeError, "changed"):
                operation()


if __name__ == "__main__":
    unittest.main()
