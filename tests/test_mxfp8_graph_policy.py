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
@dataclass
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
        self.cg=types.ModuleType('vllm.v1.worker.gpu.cudagraph_utils');self.cg.BatchExecutionDescriptor=Descriptor
        self.modules={self.comp.__name__:self.comp,self.cg.__name__:self.cg}
        self.patch=patch.dict(sys.modules,self.modules);self.patch.start()
        self.policy=load()
    def tearDown(self):self.patch.stop()

    def test_exact2048_and_small_decode_preserve_descriptor(self):
        for mode,tokens in ((Mode.PIECEWISE,2048),(Mode.FULL,32),(Mode.FULL,128)):
            desc=Descriptor(mode,tokens,32)
            self.assertIs(self.policy.apply_descriptor(desc,tokens,32),desc)
        with self.assertRaisesRegex(RuntimeError,'PIECEWISE2048'):
            self.policy.apply_descriptor(Descriptor(Mode.NONE,2048,32),2048,32)

    def test_non_target_large_tail_uses_none_without_padding(self):
        for tokens in (264,1992,2047,2056):
            desc=self.policy.apply_descriptor(Descriptor(Mode.PIECEWISE,2048,None,2),tokens,7)
            self.assertEqual(desc,Descriptor(Mode.NONE,tokens,7,2))
        with self.assertRaisesRegex(RuntimeError,'padding'):
            self.policy.apply_descriptor(Descriptor(Mode.NONE,2048,7),1992,7)
        with self.assertRaisesRegex(RuntimeError,'counts'):
            self.policy.apply_descriptor(Descriptor(Mode.NONE,1992,7),None,7)

    def test_install_once_and_no_routing_change_before_capture(self):
        class Manager:
            def dispatch(self,num_reqs,num_tokens):return Descriptor(Mode.PIECEWISE,2048,None)
        self.cg.CudaGraphManager=Manager
        self.assertTrue(self.policy.install_dispatch());self.assertFalse(self.policy.install_dispatch())
        manager=Manager()
        self.assertEqual(manager.dispatch(2,1992).num_tokens,2048)
        self.policy._READY=True
        self.assertEqual(manager.dispatch(num_reqs=2,num_tokens=1992).num_tokens,1992)
        self.assertEqual(manager.dispatch(2,2048).cg_mode,Mode.PIECEWISE)

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
        with patch.dict(sys.modules,{package.__name__:package,native.__name__:native}):
            self.assertTrue(self.policy.verify_capture(worker)['ready'])
            worker.vllm_config.cache_config.kv_cache_memory_bytes=4*2**30
            with self.assertRaisesRegex(RuntimeError,'19 GiB'):self.policy.verify_capture(worker)
            worker.vllm_config.cache_config.kv_cache_memory_bytes=19*2**30
            native.inspect=lambda worker:{'counts':{'capture_cache_miss':1}}
            with self.assertRaisesRegex(RuntimeError,'workspace'):self.policy.verify_capture(worker)

if __name__=='__main__':unittest.main()
