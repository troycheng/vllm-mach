"""CPU contracts for the shape-limited champion dual and BA integration."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import types
import unittest

import test_mxfp8_native_cpu as native_tests


class ProfileGlueTests(unittest.TestCase):
    setUp = native_tests.PackageGlueTests.setUp
    tearDown = native_tests.PackageGlueTests.tearDown
    make_backend = native_tests.PackageGlueTests.make_backend

    def dual_fixture(self):
        native, torch, Tensor, _, _, _, _ = self.make_backend()
        torch.uint8 = "u8"
        native._INSTALLED = True
        native._POLICY = "all"
        module = types.ModuleType("vllm.model_executor.kernels.linear.mxfp8.flashinfer")
        class Cutlass:
            apply_weights = native._apply_weights
        module.FlashInferCutlassMxfp8LinearKernel = Cutlass
        sys.modules[module.__name__] = module
        distributed = types.ModuleType("vllm.distributed")
        distributed.get_tensor_model_parallel_world_size = lambda: 1
        sys.modules[distributed.__name__] = distributed
        registrations = []
        def register(**kwargs):
            registrations.append(kwargs)
            setattr(torch.ops.vllm, kwargs["op_name"], kwargs["op_func"])
        utils = types.ModuleType("vllm.utils.torch_utils")
        utils.direct_register_custom_op = register
        sys.modules[utils.__name__] = utils
        native.verify_runtime = lambda **kwargs: None
        library = Path("/tmp/test-only-native.so").resolve()
        native._NATIVE_IDENTITY = {"path": str(library)}
        api = types.ModuleType("mxfp6.mxfp8_dual")
        api.load_library = lambda: library
        sys.modules[api.__name__] = api
        dual = native_tests.load("vllm_mach.mxfp8.dual", "dual.py")
        calls = []
        def quantize(x):
            calls.append(("quant", x.shape[0]))
            return "hi", "s_hi", "res", "s_res"
        def gemm(*args):
            calls.append(("dual", args))
        api.quantize = quantize
        api.gemm_out = gemm
        dual._quant_native = lambda x, w, s: calls.append(("native", x.shape[0])) or "native"
        return dual, native, torch, Tensor, Cutlass, registrations, calls

    def test_dual_routes_exact_runtime_rows_and_preserves_op_signatures(self):
        dual, _, torch, Tensor, Cutlass, registrations, calls = self.dual_fixture()
        self.assertTrue(dual.install())
        self.assertFalse(dual.install())
        self.assertEqual([item["op_name"] for item in registrations],
                         ["mach_mx8_dual_gateup", "mach_mx8_dual_qkvz_both"])
        qkvz = types.SimpleNamespace(weight=Tensor((12288, 2560), "e4m3"),
                                    weight_scale=Tensor((12288, 80), "u8"),
                                    _mx8_dual_qkvz_both=True, _mx8_dual_qkvz_both_id=3)
        gateup = types.SimpleNamespace(weight=Tensor((18432, 2560), "e4m3"),
                                      weight_scale=Tensor((18432, 80), "u8"),
                                      _mx8_dual_gateup=True, _mx8_dual_gateup_id=7)
        for m in (32, 64):
            out = Cutlass().apply_weights(qkvz, Tensor((m, 2560)))
            self.assertEqual(out.shape, (m, 12288))
        self.assertEqual(Cutlass().apply_weights(gateup, Tensor((32, 2560))).shape,
                         (32, 18432))
        self.assertEqual(Cutlass().apply_weights(gateup, Tensor((64, 2560))), "native")
        self.assertEqual(Cutlass().apply_weights(qkvz, Tensor((16, 2560))), "native")
        self.assertEqual([call[1] for call in calls if call[0] == "quant"], [32, 64, 32])
        self.assertEqual([call[1] for call in calls if call[0] == "native"], [64, 16])
        args = next(call[1] for call in calls if call[0] == "dual")
        self.assertEqual(args[:3], ("hi", qkvz.weight, "s_hi"))
        self.assertEqual(args[4:6], ("res", "s_res"))
        torch.cuda.is_current_stream_capturing = lambda: True
        Cutlass().apply_weights(qkvz, Tensor((64, 2560)))
        self.assertEqual(dual._CAPTURE_LAYERS[64][3], 1)

    def test_dual_install_rejects_wrong_predecessor_tp_and_library(self):
        dual, native, _, _, Cutlass, registrations, _ = self.dual_fixture()
        native._POLICY = "wide-only"
        with self.assertRaisesRegex(RuntimeError, "policy=all"):
            dual.install()
        native._POLICY = "all"
        sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 2
        with self.assertRaisesRegex(RuntimeError, "parallel size 1"):
            dual.install()
        sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
        Cutlass.apply_weights = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "immediate MLP32 predecessor"):
            dual.install()
        Cutlass.apply_weights = native._apply_weights
        sys.modules["mxfp6.mxfp8_dual"].load_library = lambda: Path("/tmp/wrong.so")
        with self.assertRaisesRegex(RuntimeError, "different native libraries"):
            dual.install()
        self.assertEqual(registrations, [])

    def test_dual_tagging_preserves_all_mlp_and_gdn_ordinal_ids(self):
        dual, _, _, Tensor, Cutlass, _, _ = self.dual_fixture()
        dual.install()
        mlp_module = types.ModuleType("vllm.model_executor.models.qwen2_moe")
        class MLP:
            expert_gate = None
        mlp_module.Qwen2MoeMLP = MLP
        sys.modules[mlp_module.__name__] = mlp_module
        gdn_module = types.ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        class GDN:
            pass
        gdn_module.QwenGatedDeltaNetAttention = GDN
        sys.modules[gdn_module.__name__] = gdn_module
        modules = []
        def layer(n):
            return types.SimpleNamespace(weight=Tensor((n, 2560), "e4m3"),
                                         weight_scale=Tensor((n, 80), "u8"),
                                         scheme=types.SimpleNamespace(kernel=Cutlass()))
        for i in range(32):
            mlp = MLP()
            mlp.gate_up_proj = layer(18432)
            modules.append((f"model.layers.{i}.mlp", mlp))
        for i in range(24):
            gdn = GDN()
            gdn.in_proj_qkvz = layer(12288)
            modules.append((f"model.layers.{i + i // 3}.linear_attn", gdn))
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
            model=types.SimpleNamespace(named_modules=lambda: modules)))
        tags = dual.tag_model(worker)
        self.assertEqual(len(tags["gateup"]), 32)
        self.assertEqual(len(tags["qkvz"]), 24)
        self.assertEqual([module.in_proj_qkvz._mx8_dual_qkvz_both_id
                          for _, module in modules if isinstance(module, GDN)], list(range(24)))
        self.assertEqual([module.gate_up_proj._mx8_dual_gateup_id
                          for _, module in modules if isinstance(module, MLP)], list(range(32)))
        with self.assertRaisesRegex(RuntimeError, "coverage missing"):
            dual.verify_capture()
        dual._MLP_CAPTURE_LAYERS.update(range(32))
        dual._CAPTURE_LAYERS[32].update(range(24))
        dual._CAPTURE_LAYERS[64].update(range(24))
        self.assertEqual(dual.verify_capture()["qkvz_layer_count"], 24)

    def test_ba_runtime_fallback_uses_original_weight_and_exact_small_m_range(self):
        _, torch, Tensor, _, _, _, _ = self.make_backend()
        torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(
            linear=lambda x, weight: ("stock", weight)))
        ba = native_tests.load("vllm_mach.mxfp8.ba", "ba.py")
        calls = []
        ba._GEMV = types.SimpleNamespace(
            bf16_gemv_small_n=lambda x, weight: calls.append((x.shape[0], weight)) or "gemv")
        original, private = Tensor((64, 2560)), Tensor((64, 2560))
        for m in (1, 2, 4, 8):
            self.assertEqual(ba._small_impl(Tensor((m, 2560)), original, private), "gemv")
        for m in (0, 9, 32, 64, 2048):
            self.assertEqual(ba._small_impl(Tensor((m, 2560)), original, private),
                             ("stock", original))
        self.assertEqual([m for m, _ in calls], [1, 2, 4, 8])
        self.assertTrue(all(weight is private for _, weight in calls))

    def test_ba_install_is_pinned_and_keeps_postload_then_clone_order(self):
        _, torch, Tensor, _, _, _, _ = self.make_backend()
        events = []
        class Linear:
            def process_weights_after_loading(self, layer):
                events.append("stock_post")
                return "loaded"
            def apply(self, layer, x, bias=None):
                return "stock_apply"
        module = types.ModuleType("vllm.model_executor.layers.linear")
        module.UnquantizedLinearMethod = Linear
        sys.modules[module.__name__] = module
        gemv = types.ModuleType("vllm_mach.mxfp8.ba_gemv")
        gemv.precompile_bf16_gemv_small_n = lambda private: events.append("precompile")
        gemv.bf16_gemv_small_n = lambda x, private: "small"
        sys.modules[gemv.__name__] = gemv
        class CustomOp:
            def __init__(self, fn):
                self.fn = fn
            def register_fake(self, fn):
                self.fake = fn
            def __call__(self, *args):
                return self.fn(*args)
        schemas = []
        def custom_op(name, *, mutates_args):
            schemas.append((name, mutates_args))
            return CustomOp
        torch.library = types.SimpleNamespace(custom_op=custom_op)
        torch.equal = lambda x, y: True
        torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(linear=lambda x, w: "linear"))
        Tensor.is_cuda = True
        Tensor.detach = lambda x: x
        Tensor.clone = lambda x: Tensor(x.shape, x.dtype)
        Tensor.contiguous = lambda x: x
        ba = native_tests.load("vllm_mach.mxfp8.ba", "ba.py")
        ba.version = lambda name: "1.5.0"
        with self.assertRaisesRegex(RuntimeError, "requires b12x 1.2.6"):
            ba.install()
        self.assertEqual(schemas, [])
        ba.version = lambda name: "1.2.6"
        self.assertTrue(ba.install())
        self.assertFalse(ba.install())
        self.assertEqual(schemas, [("mx8_ba_only::small_n", ())])
        # The basic fake Tensor stores contiguous as a flag; supply a genuine
        # clone method's chain for the weight preparation fixture.
        class Weight(Tensor):
            def clone(self):
                return types.SimpleNamespace(contiguous=lambda: Tensor(self.shape, self.dtype))
        layer = types.SimpleNamespace(weight=Weight((64, 2560)), bias=None)
        self.assertEqual(Linear().process_weights_after_loading(layer), "loaded")
        self.assertEqual(events, ["stock_post", "precompile"])
        self.assertIsNot(layer._mx8_ba_private_weight, layer.weight)
        self.assertEqual(Linear().apply(layer, Tensor((4, 2560))), "small")
        self.assertEqual(Linear().apply(layer, Tensor((4, 2560)), object()), "stock_apply")
        with self.assertRaisesRegex(RuntimeError, "prepared twice"):
            Linear().process_weights_after_loading(layer)

    def test_reused_gemv_kernel_and_launch_source_are_unchanged(self):
        # Hash reviewed function/class source segments without importing the
        # CUDA compiler. The release test is self-contained, with no old run
        # directory or experiment fixture dependency.
        current = native_tests.HERE / "mxfp8/ba_gemv.py"
        source = current.read_text()
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        body = "\n\n".join(ast.get_source_segment(source, node) for node in nodes)
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
                         "a467b7d38b670168791e1e84a1a9126976faf3071377f088ba24e37021eb0c6e")


if __name__ == "__main__":
    unittest.main()
