"""CPU routing/capture contract for exact2048 without importing CUDA libraries."""
import enum
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / 'src/vllm_mach/mxfp8/graph_policy.py'
class Mode(enum.Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    FULL_AND_PIECEWISE = 3
@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int | None
    num_active_loras: int = 0

def load():
    spec=importlib.util.spec_from_file_location('vllm_mach.mxfp8.graph_policy',FILE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class GraphPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.comp=types.ModuleType('vllm.config.compilation');self.comp.CUDAGraphMode=Mode
        self.modules={self.comp.__name__:self.comp}
        self.patch=patch.dict(sys.modules,self.modules);self.patch.start()
        self.policy=load()
    def tearDown(self):self.patch.stop()

    def test_capture_policy_capacity_and_native_workspace_verification(self):
        native=types.ModuleType('vllm_mach.mxfp8.native_backend')
        native.inspect=lambda worker:{'counts':{'capture_cache_miss':0}}
        package=types.ModuleType('vllm_mach.mxfp8');package.__path__=[];package.native_backend=native
        worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(cudagraph_manager=types.SimpleNamespace(
            _graphs_captured=True,_capture_descs={
                Mode.PIECEWISE:[Descriptor(Mode.PIECEWISE,x,None) for x in self.policy.CAPTURE_SIZES],
                Mode.FULL:[Descriptor(Mode.FULL,x,x) for x in self.policy.FULL_SIZES]})),
            vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(
                cudagraph_mode=Mode.FULL_AND_PIECEWISE,max_cudagraph_capture_size=2048),
                cache_config=types.SimpleNamespace(kv_cache_memory_bytes=19*2**30)))
        manager = worker.model_runner.cudagraph_manager
        manager.dispatch = lambda **kwargs: Descriptor(Mode.PIECEWISE, 2048, None)
        wrapper = types.SimpleNamespace(
            vllm_config=worker.vllm_config, runtime_mode=Mode.PIECEWISE,
            concrete_cudagraph_entries={Descriptor(Mode.PIECEWISE,2048,None):
                                        types.SimpleNamespace(cudagraph=object())})
        module = types.ModuleType("vllm.compilation.cuda_graph")
        module.CUDAGraphWrapper = types.SimpleNamespace(_all_instances=[wrapper])
        with patch.dict(sys.modules,{package.__name__:package,native.__name__:native,
                                     module.__name__:module}):
            self.assertEqual(self.policy.verify_capture(worker)["dispatch"], "vllm")
            with patch.object(manager, "dispatch", side_effect=lambda **kw:
                    Descriptor(Mode.NONE, kw["num_tokens"], 1) if kw["num_tokens"] == 2040
                    else Descriptor(Mode.PIECEWISE, 2048, None)):
                with self.assertRaisesRegex(RuntimeError, "prefill replay at 2040"):
                    self.policy.verify_capture(worker)
            with patch.object(module.CUDAGraphWrapper, "_all_instances", []):
                with self.assertRaisesRegex(RuntimeError, "prefill capture"):
                    self.policy.verify_capture(worker)
            worker.vllm_config.cache_config.kv_cache_memory_bytes=4*2**30
            with self.assertRaisesRegex(RuntimeError,'19 GiB'):self.policy.verify_capture(worker)
            worker.vllm_config.cache_config.kv_cache_memory_bytes=19*2**30
            native.inspect=lambda worker:{'counts':{'capture_cache_miss':1}}
            with self.assertRaisesRegex(RuntimeError,'workspace'):self.policy.verify_capture(worker)

if __name__=='__main__':unittest.main()
