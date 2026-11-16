"""CPU-only binding, inverse fallback and frozen-source checks."""
import ast
import hashlib
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src"
if not os.environ.get("MACH_TEST_RUNTIME_PROFILE"):
    sys.path.insert(0, str(SOURCE))
from vllm_mach.mxfp8.two_b import head_scale as mod

FROZEN_AST = {'scalar_finalize': '26f272f155c195ecbc00aac95e06fd5b7c88cf764306334a233ac8e6eff635e8', 'reduce_scale': '751719b46ce0bde8927e5bd7e710b39507f15eec60834bb3e150aebea681c50c', 'eligible': 'bc575e4d97cd00150de5c85f782aff28d2e3fdd9312caa3feffeb991c2126367', 'guarded': '8f9330b7e3ea8adc31422684244cebe61b77c0173b998cf4761cbcc5413ded67'}


def worker():
    return NS(_mach_2b_contract=dict(mode="production", ssm_dtype="float32", hybrid_greedy_head=True),
              _mach_2b_hybrid_head={"state": NS(candidates=128, _head_graphs=None)},
              vllm_config=NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)),
              model_runner=NS(speculative_config=None, cudagraph_manager=NS(_graphs_captured=True, graphs={16: object()})))


def original_module():
    path = Path(mod.__file__).resolve().parents[2] / "mxfp6/hybrid_nvfp4_lm_head.py"
    module = types.ModuleType("head_scale_owner_fixture")
    module.__file__ = str(path)
    code = compile(path.read_text(), str(path), "exec", dont_inherit=True)
    original = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "_global_scale")
    module._global_scale = types.FunctionType(original, vars(module))
    exec("def coarse_logits(self): pass", vars(module))
    module.HybridNvfp4LmHead = NS(coarse_logits=module.coarse_logits)
    module._NVFP4_MAX_VALUE = 2688.0
    return module


def guard_fixture():
    flags = dict(inference=True, grad=False, compile=False, trace=False, script=False,
                 capture=False, dispatch=None, function=None)
    class Tensor:
        is_cuda = True
        dtype = "bf16"
        shape = (16, 2048)
        device = "cuda:0"
        def stride(self): return (2048, 1)
        def is_conj(self): return False
        def is_neg(self): return False
    outputs, launches = [], []
    def empty(*args, **kwargs):
        value = object(); outputs.append((value, args, kwargs)); return value
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))
    torch = NS(Tensor=Tensor, bfloat16="bf16", float32="fp32", empty=empty,
               is_inference_mode_enabled=lambda: flags["inference"], is_grad_enabled=lambda: flags["grad"],
               compiler=NS(is_compiling=lambda: flags["compile"]),
               jit=NS(is_tracing=lambda: flags["trace"], is_scripting=lambda: flags["script"]),
               cuda=NS(is_current_stream_capturing=lambda: flags["capture"]),
               _is_zerotensor=lambda tensor: getattr(tensor, "zero", False))
    modules = {"torch.utils._python_dispatch": NS(_get_current_dispatch_mode=lambda: flags["dispatch"]),
               "torch.overrides": NS(_get_current_function_mode=lambda: flags["function"]),
               "vllm_mach.mxfp8.two_b._head_scale_kernel": NS(reduce_scale=Kernel())}
    return torch, Tensor, flags, outputs, launches, modules


class HeadScaleTests(unittest.TestCase):
    def test_frozen_kernel_guard_and_launch_AST(self):
        nodes = {}
        for name in ("head_scale.py", "_head_scale_kernel.py"):
            for node in ast.walk(ast.parse(Path(mod.__file__).with_name(name).read_text())):
                if isinstance(node, ast.FunctionDef) and node.name in FROZEN_AST:
                    nodes[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(nodes, FROZEN_AST)
        self.assertNotIn("scalar_probe_kernel", Path(mod.__file__).with_name("_head_scale_kernel.py").read_text())

    def test_admitted_launch_fresh_scalar_and_original_inverse(self):
        torch, Tensor, flags, outputs, launches, modules = guard_fixture()
        calls = []; result = object(); error = RuntimeError("original")
        def reference(tensor):
            calls.append(tensor)
            if getattr(tensor, "raise_error", False): raise error
            return result
        with patch.dict(sys.modules, modules):
            guarded = mod._bind(torch, reference)
        tensor = Tensor()
        self.assertIs(guarded(tensor), outputs[0][0])
        self.assertIs(guarded(tensor), outputs[1][0])
        self.assertIsNot(outputs[0][0], outputs[1][0])
        self.assertEqual(calls, [])
        self.assertEqual(launches[0], ((1,), (tensor, outputs[0][0]), dict(num_warps=8, enable_fp_fusion=False)))
        self.assertEqual(outputs[0][1:], (((),), dict(dtype="fp32", device="cuda:0")))
        rejected = []
        for key, value in (("shape", (15,2048)), ("shape", (17,2048)), ("shape", (16,1024)),
                           ("dtype", "fp32"), ("is_cuda", False), ("stride", lambda: (1,16)),
                           ("is_conj", lambda: True), ("is_neg", lambda: True), ("zero", True)):
            value_tensor = Tensor(); setattr(value_tensor, key, value); rejected.append(value_tensor)
        class Subclass(Tensor): pass
        rejected.append(Subclass())
        for value_tensor in rejected:
            self.assertIs(guarded(value_tensor), result)
            self.assertIs(calls[-1], value_tensor)
        for key in flags:
            old = flags[key]; flags[key] = False if key == "inference" else True
            self.assertIs(guarded(tensor), result); flags[key] = old
        self.assertEqual(len(calls), len(rejected) + len(flags))
        self.assertEqual(len(launches), 2)
        tensor.shape = (1,2048); tensor.raise_error = True
        with self.assertRaises(RuntimeError) as caught: guarded(tensor)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), len(rejected) + len(flags) + 1)

    def test_selected_install_duplicate_and_owner_refusal(self):
        selected = worker(); module = original_module(); original = module._global_scale
        sentinel = object(); torch = NS(cuda=NS(is_current_stream_capturing=lambda: False))
        self.assertIs(mod._original(module), original)
        with patch.object(mod, "_runtime", return_value=(torch, module)), patch.object(mod, "_bind", return_value=sentinel) as bind:
            self.assertTrue(mod.install(selected))
            bind.assert_called_once_with(torch, original)
            self.assertIs(module._global_scale, sentinel)
            self.assertTrue(mod.inspect_worker(selected)["installed"])
            with self.assertRaisesRegex(RuntimeError, "already installed"): mod.install(selected)
            with self.assertRaisesRegex(RuntimeError, "owner changed"): mod.install(worker())
        for mutation in (lambda m: setattr(m, "_NVFP4_MAX_VALUE", 1),
                         lambda m: setattr(m, "_global_scale", lambda x: x),
                         lambda m: setattr(m, "HybridNvfp4LmHead", NS(coarse_logits=lambda x: x))):
            module = original_module(); mutation(module)
            with self.assertRaisesRegex(RuntimeError, "owner changed"): mod._original(module)
        module = original_module(); module._global_scale.__code__ = (lambda x: x).__code__
        with self.assertRaisesRegex(RuntimeError, "source owner changed"): mod._original(module)

    def test_unselected_profiles_leave_runtime_and_original_untouched(self):
        rejected = [NS()]
        for key,value in (("mode","quality"),("ssm_dtype","bfloat16"),("hybrid_greedy_head",False)):
            value_worker = worker(); value_worker._mach_2b_contract[key] = value; rejected.append(value_worker)
        for key,value in (("candidates",64),("_head_graphs",object())):
            value_worker = worker(); setattr(value_worker._mach_2b_hybrid_head["state"],key,value); rejected.append(value_worker)
        for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
            value_worker=worker(); setattr(value_worker.vllm_config.parallel_config,key,2); rejected.append(value_worker)
        value_worker=worker(); value_worker.model_runner.speculative_config=object(); rejected.append(value_worker)
        value_worker=worker(); value_worker.model_runner.cudagraph_manager._graphs_captured=False; rejected.append(value_worker)
        value_worker=worker(); value_worker.model_runner.cudagraph_manager.graphs={}; rejected.append(value_worker)
        with patch.object(mod,"_runtime",side_effect=AssertionError("unselected runtime import")):
            for value_worker in rejected: self.assertFalse(mod.install(value_worker))

    def test_capture_install_refuses_before_binding(self):
        module=original_module(); original=module._global_scale
        with patch.object(mod,"_runtime",return_value=(NS(cuda=NS(is_current_stream_capturing=lambda: True)),module)):
            with self.assertRaisesRegex(RuntimeError,"completed capture"): mod.install(worker())
        self.assertIs(module._global_scale,original)

    def test_actual_CPU_tensor_falls_back_without_CUDA_initialization(self):
        try: import torch
        except ImportError: self.skipTest("PyTorch unavailable on local host")
        kernel = NS(reduce_scale=object()); result=object(); calls=[]
        with patch.dict(sys.modules,{"vllm_mach.mxfp8.two_b._head_scale_kernel":kernel}):
            guarded=mod._bind(torch,lambda tensor: calls.append(tensor) or result)
        tensor=torch.zeros((16,2048),dtype=torch.bfloat16)
        with torch.inference_mode(): self.assertIs(guarded(tensor),result)
        self.assertEqual(len(calls),1)
        self.assertIs(calls[0],tensor)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
