"""CPU checks of 2B BA integration and graph ownership/capacity contracts."""
import ast
import hashlib
from dataclasses import dataclass
from enum import Enum, auto
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/two_b"
NS = types.SimpleNamespace
PROFILE = "qwen35-2b-mxfp8-champion-v1"


def load(name):
    spec = importlib.util.spec_from_file_location("cpu_2b_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mode(Enum):
    NONE = auto()
    FULL = auto()
    PIECEWISE = auto()
    FULL_AND_PIECEWISE = auto()


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int = 1
    num_active_loras: int = 0


def graph_api():
    class Manager:
        def dispatch(self, num_reqs, num_tokens, uniform_token_count=None, **kwargs):
            route = Mode.FULL if uniform_token_count == 1 else Mode.PIECEWISE
            available = sorted(d.num_tokens for d in self._capture_descs[route])
            padded = next((m for m in available if m >= num_tokens), None)
            return Descriptor(route if padded else Mode.NONE,
                              padded if padded else num_tokens, num_reqs,
                              kwargs.get("num_active_loras", 0))
    class Wrapper:
        _all_instances = []
    sentinel = object()
    modules = {
        "vllm.config.compilation": NS(CUDAGraphMode=Mode),
        "vllm.compilation.cuda_graph": NS(CUDAGraphWrapper=Wrapper),
        "vllm.v1.worker.gpu.kv_connector": NS(NO_OP_KV_CONNECTOR=sentinel),
        "vllm.v1.worker.gpu.cudagraph_utils": NS(
            BatchExecutionDescriptor=Descriptor, CudaGraphManager=Manager)}
    return modules, Manager, Wrapper, sentinel


def worker_fixture(Manager, Wrapper, sentinel, *, rows=None):
    full = [4, 8, 16, 24, 32, 48, 64, 96, 128, 160] if rows is None else [m for m in (4, 32, 64) if m <= rows]
    pw = full + [2048] if rows is None else full
    cfg = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
             lora_config=None, speculative_config=None,
             model_config=NS(max_model_len=16384 if rows is None else 1024),
             scheduler_config=NS(max_num_seqs=160 if rows is None else rows,
                                 max_num_batched_tokens=2048 if rows is None else max(8192, rows*256)),
             cache_config=NS(kv_cache_memory_bytes=(19 if rows is None else 4)*2**30,
                             enable_prefix_caching=False),
             compilation_config=NS(cudagraph_mode=Mode.FULL_AND_PIECEWISE,
                                   cudagraph_capture_sizes=pw, max_cudagraph_capture_size=pw[-1]))
    manager = Manager()
    manager._graphs_captured, manager.use_breakable_cg = True, False
    manager.cudagraph_mode = Mode.FULL_AND_PIECEWISE
    manager._capture_descs = {Mode.FULL: [Descriptor(Mode.FULL, m) for m in full],
                              Mode.PIECEWISE: [Descriptor(Mode.PIECEWISE, m) for m in pw]}
    manager.graphs = {d: object() for d in manager._capture_descs[Mode.FULL]}
    Wrapper._all_instances = [NS(vllm_config=cfg, runtime_mode=Mode.PIECEWISE,
        concrete_cudagraph_entries={d: NS(cudagraph=object()) for d in manager._capture_descs[Mode.PIECEWISE]})]
    return NS(vllm_config=cfg, model_runner=NS(cudagraph_manager=manager, kv_connector=sentinel))


class TwoBGraphContracts(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"VLLM_MACH_PROFILE": PROFILE}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_exact2048_and_unpadded_tails_use_installed_dispatch(self):
        policy = load("graph_policy")
        modules, Manager, Wrapper, sentinel = graph_api()
        with patch.dict(sys.modules, modules):
            self.assertTrue(policy.install_dispatch())
            self.assertFalse(policy.install_dispatch())
            worker = worker_fixture(Manager, Wrapper, sentinel)
            receipt = policy.verify_capture(worker)
            self.assertEqual(receipt["kv_capacity_bytes"], 19*2**30)
            manager = worker.model_runner.cudagraph_manager
            for tokens in (161, 512, 2047, 2049, 8192):
                desc = manager.dispatch(3, tokens, num_active_loras=2)
                self.assertEqual(desc, Descriptor(Mode.NONE, tokens, 3, 2))
            self.assertEqual(manager.dispatch(1, 2048).cg_mode, Mode.PIECEWISE)
            self.assertEqual(manager.dispatch(5, 5, uniform_token_count=1).num_tokens, 8)
            with self.assertRaisesRegex(RuntimeError, "exact PIECEWISE2048"):
                policy.apply_descriptor(Descriptor(Mode.FULL, 2048), 2048, 1)
            with self.assertRaisesRegex(RuntimeError, "exact PIECEWISE2048"):
                policy.apply_descriptor(Descriptor(Mode.PIECEWISE, 4096), 2048, 1)

    def test_plan_and_flag_do_not_prove_real_graph_capture(self):
        for kind in ("full_missing", "full_none", "pw_none", "pw_missing", "pw_wrong_owner"):
            policy = load("graph_policy")
            modules, Manager, Wrapper, sentinel = graph_api()
            with patch.dict(sys.modules, modules):
                policy.install_dispatch()
                worker = worker_fixture(Manager, Wrapper, sentinel)
                manager = worker.model_runner.cudagraph_manager
                first = next(iter(manager.graphs))
                if kind == "full_missing": del manager.graphs[first]
                elif kind == "full_none": manager.graphs[first] = None
                elif kind == "pw_none": next(iter(Wrapper._all_instances[0].concrete_cudagraph_entries.values())).cudagraph = None
                elif kind == "pw_missing": Wrapper._all_instances.clear()
                else: Wrapper._all_instances[0].vllm_config = object()
                with self.assertRaisesRegex(RuntimeError, "graph capture incomplete"):
                    policy.verify_capture(worker)
                self.assertFalse(hasattr(manager, "_mach_2b_graph_mode"))

    def test_capacity_and_runtime_rejections(self):
        policy = load("graph_policy")
        modules, Manager, Wrapper, sentinel = graph_api()
        with patch.dict(sys.modules, modules):
            policy.install_dispatch()
            for section, field, value in (("cache_config", "kv_cache_memory_bytes", 4*2**30),
                                         ("cache_config", "enable_prefix_caching", True),
                                         ("scheduler_config", "max_num_seqs", 128),
                                         ("scheduler_config", "max_num_batched_tokens", 4096),
                                         ("model_config", "max_model_len", 8192),
                                         ("parallel_config", "tensor_parallel_size", 2)):
                worker = worker_fixture(Manager, Wrapper, sentinel)
                setattr(getattr(worker.vllm_config, section), field, value)
                with self.assertRaisesRegex(RuntimeError, "capacity/runtime"):
                    policy.verify_capture(worker)
            worker = worker_fixture(Manager, Wrapper, sentinel)
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 2048]
            with self.assertRaisesRegex(RuntimeError, "plan differs"):
                policy.verify_capture(worker)

    def test_quality_fixed_captures_capacity_and_uninstalled_production_policy(self):
        policy = load("graph_policy")
        modules, Manager, Wrapper, sentinel = graph_api()
        with patch.dict(sys.modules, modules):
            for rows, sizes in ((4, [4]), (32, [4, 32]), (64, [4, 32, 64])):
                with patch.dict(os.environ, {"VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": str(rows)}):
                    self.assertEqual(policy.compilation_settings()["cudagraph_capture_sizes"], sizes)
                    worker = worker_fixture(Manager, Wrapper, sentinel, rows=rows)
                    result = policy.verify_capture(worker)
                    self.assertEqual(result["kv_capacity_bytes"], 4*2**30)
                    self.assertEqual(result["full_sizes"], sizes)
                    worker.vllm_config.cache_config.kv_cache_memory_bytes = 19*2**30
                    with self.assertRaisesRegex(RuntimeError, "capacity/runtime"):
                        policy.verify_capture(worker)
            for env in ({"VLLM_MACH_MXFP8_MODE": "quality"}, {"VLLM_MACH_MXFP8_MODE": "unknown"},
                        {"VLLM_MACH_MXFP8_QUALITY_ROWS": "4"}):
                with patch.dict(os.environ, env):
                    with self.assertRaises(RuntimeError): policy.compilation_settings()


class Tensor:
    def __init__(self, shape=(32, 2048), dtype="bf16", cuda=True, pointer=16, contiguous=True):
        self.shape, self.dtype, self.is_cuda = shape, dtype, cuda
        self.pointer, self.contiguous_flag = pointer, contiguous
        self.ndim, self.device = len(shape), "cuda:0"
    def detach(self): return self
    def clone(self): return Tensor(self.shape, self.dtype, self.is_cuda, self.pointer+16)
    def contiguous(self): self.contiguous_flag = True; return self
    def is_contiguous(self): return self.contiguous_flag
    def data_ptr(self): return self.pointer
    def view(self, dtype): return self
    def numel(self): return self.shape[0]*self.shape[1]
    def element_size(self): return 2


def ba_api(events):
    class Method:
        def process_weights_after_loading(self, layer): events.append("parent_post")
        def apply(self, layer, x, bias=None): events.append("parent_apply"); return "stock"
    class Op:
        def __init__(self, fn): self.fn = fn
        def register_fake(self, fn): self.fake = fn
        def __call__(self, *args): return self.fn(*args)
    torch = NS(Tensor=Tensor, bfloat16="bf16", uint8="u8", equal=lambda a,b: a.shape == b.shape,
               library=NS(custom_op=lambda name, **kw: lambda fn: Op(fn)),
               cuda=NS(is_current_stream_capturing=lambda: True),
               empty=lambda shape, **kw: Tensor(shape),
               nn=NS(functional=NS(linear=lambda x,w: "fallback")))
    cache = {}
    def precompile(weight):
        events.append("precompile")
        for m in range(1, 9): cache[m] = lambda x,w,y: events.append(("launch", x.shape[0]))
    kernel = NS(precompile_bf16_gemv_small_n=precompile,
                get_cached_bf16_gemv_small_n=lambda m,n,k: cache.get(m))
    return {"torch": torch, "vllm.model_executor.layers.linear": NS(UnquantizedLinearMethod=Method)}, Method, kernel, cache


class TwoBBAContracts(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"VLLM_MACH_PROFILE": PROFILE}, clear=True)
        env.start(); self.addCleanup(env.stop)

    def test_profile_guard_and_audited_kernel_resolution(self):
        ba = load("ba")
        self.assertTrue(ba.enabled())
        with patch.dict(os.environ, {"VLLM_MACH_PROFILE": "other"}):
            self.assertFalse(ba.enabled())
            with self.assertRaisesRegex(RuntimeError, "requires qwen35-2b"):
                ba.install()
        with patch.object(ba, "version", return_value="1.2.6"), \
                patch.object(ba, "import_module", return_value=NS(__file__=ROOT.parent/"ba_gemv.py")):
            self.assertEqual(ba.load_kernel().__file__, ROOT.parent/"ba_gemv.py")
        with patch.object(ba, "version", return_value="1.2.7"):
            with self.assertRaisesRegex(RuntimeError, "audited b12x"):
                ba.load_kernel()
        self.assertFalse(hasattr(ba, "install_worker_hook"))
        self.assertFalse(hasattr(ba, "install_fixture_capture"))

    def test_clone_prewarms_all_rows_then_opaque_launch_and_guards(self):
        ba = load("ba")
        events = []
        modules, Method, kernel, cache = ba_api(events)
        with patch.dict(sys.modules, modules), patch.object(ba, "load_kernel", return_value=kernel), \
                patch.object(ba, "version", return_value="0.29.0"):
            self.assertTrue(ba.install())
            method, layer = Method(), NS(weight=Tensor(), bias=None)
            method.process_weights_after_loading(layer)
            self.assertEqual(events, ["parent_post", "precompile"])
            self.assertEqual(set(cache), set(range(1, 9)))
            self.assertIsNot(layer._two_b_ba_private_weight, layer.weight)
            for m in range(1, 9):
                y = method.apply(layer, Tensor((m, 2048)))
                self.assertEqual(y.shape, (m, 32))
                self.assertEqual(ba._COUNTS[m]["capture_kernel_calls"], 1)
            self.assertEqual(method.apply(layer, Tensor((4, 1024))), "stock")
            self.assertEqual(method.apply(layer, Tensor((4, 2048), dtype="f32")), "stock")
            self.assertEqual(method.apply(layer, Tensor((4, 2048), cuda=False)), "stock")
            self.assertEqual(method.apply(layer, Tensor((4, 2048)), bias=object()), "stock")
            self.assertEqual(method.apply(layer, Tensor((9, 2048))), "fallback")
            cache.pop(4)
            self.assertEqual(method.apply(layer, Tensor((4, 2048))), "fallback")
            self.assertEqual(ba._COUNTS[4]["capture_fallback_cache_miss"], 1)
            with self.assertRaisesRegex(RuntimeError, "prepared twice"):
                method.process_weights_after_loading(layer)

    def test_all_18_clones_are_bitwise_checked_and_shape_limited(self):
        ba = load("ba")
        events = []
        modules, Method, kernel, cache = ba_api(events)
        kernel.__file__ = ROOT.parent/"ba_gemv.py"
        with patch.dict(sys.modules, modules), patch.object(ba, "load_kernel", return_value=kernel), \
                patch.object(ba, "version", return_value="0.29.0"):
            ba.install()
            method = Method()
            layers = []
            for index in range(18):
                layer = NS(weight=Tensor(), bias=None)
                method.process_weights_after_loading(layer)
                layers.append((f"model.layers.{index}.in_proj_ba", layer))
            ignored = NS(weight=Tensor((64, 2048)), bias=None)
            method.process_weights_after_loading(ignored)
            self.assertFalse(hasattr(ignored, "_two_b_ba_private_weight"))
            worker = NS(model_runner=NS(model=NS(named_modules=lambda: layers)))
            receipt = ba.inspect_worker(worker)
            self.assertEqual(receipt["ba_layer_count"], 18)
            self.assertEqual(receipt["clone_bytes"], 18*32*2048*2)
            self.assertEqual(receipt["precompile_invocations"], 18)
            private = layers[0][1]._two_b_ba_private_weight
            private.pointer = layers[0][1].weight.pointer
            with self.assertRaisesRegex(RuntimeError, "invalid/changed private"):
                ba.inspect_worker(worker)
            private.pointer += 16
            layers.pop()
            with self.assertRaisesRegex(RuntimeError, "18 unique clones"):
                ba.inspect_worker(worker)

    def test_capture_weight_and_input_alignment_guards(self):
        ba = load("ba")
        events = []
        modules, Method, kernel, cache = ba_api(events)
        with patch.dict(sys.modules, modules), patch.object(ba, "load_kernel", return_value=kernel), \
                patch.object(ba, "version", return_value="0.29.0"):
            ba.install()
            layer = NS(weight=Tensor(), bias=None)
            Method().process_weights_after_loading(layer)
            x = Tensor((4, 2048), contiguous=False)
            self.assertEqual(Method().apply(layer, x).shape, (4, 32))
            self.assertEqual(ba._COUNTS[4]["capture_input_contiguous_copies"], 1)
            self.assertEqual(Method().apply(layer, Tensor((4, 2048), pointer=17)), "fallback")
            self.assertEqual(ba._COUNTS[4]["capture_fallback_input_alignment"], 1)
            layer._two_b_ba_private_weight.dtype = "f32"
            self.assertEqual(Method().apply(layer, Tensor((4, 2048))), "fallback")
            self.assertEqual(ba._COUNTS[4]["capture_fallback_weight"], 1)

    def test_missing_prewarm_rejects_prepare_before_binding_clone(self):
        ba = load("ba")
        events=[]
        modules, Method, kernel, cache = ba_api(events)
        kernel.precompile_bf16_gemv_small_n = lambda weight: None
        with patch.dict(sys.modules, modules), patch.object(ba, "load_kernel", return_value=kernel), \
                patch.object(ba, "version", return_value="0.29.0"):
            ba.install()
            layer = NS(weight=Tensor(), bias=None)
            with self.assertRaisesRegex(RuntimeError, "missed a row count"):
                Method().process_weights_after_loading(layer)
            self.assertFalse(hasattr(layer, "_two_b_ba_private_weight"))

    def test_capture_receipt_rejects_missing_rows_or_fallback(self):
        ba = load("ba")
        with patch.object(ba, "inspect_worker", return_value={"installed": True}):
            ba._COUNTS = {4: {"capture_kernel_calls": 18}}
            with self.assertRaisesRegex(RuntimeError, "missing=\\[8\\]"):
                ba.verify_capture(object(), required_rows=(4,8))
            self.assertTrue(ba.verify_capture(object(), required_rows=(4,))["installed"])
            ba._COUNTS[4]["capture_fallback_cache_miss"] = 1
            with self.assertRaisesRegex(RuntimeError, "small-M fallback"):
                ba.verify_capture(object(), required_rows=(4,))

    def test_kernel_launch_body_remains_frozen(self):
        tree = ast.parse((ROOT/"ba.py").read_text())
        launch = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_small_impl")
        # Frozen source AST digest excludes formatting but pins the launch body.
        self.assertEqual(hashlib.sha256(ast.dump(launch).encode()).hexdigest(),
                         "7660d68ff55b431ea3e5b055bcfd1277d47e41c7f5207520058a440214341161")


if __name__ == "__main__":
    unittest.main()
