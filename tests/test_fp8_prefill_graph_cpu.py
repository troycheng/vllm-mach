"""Finite Block2048 production scope and original-wrapper lifecycle."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

import pytest
from vllm_mach.fp8 import prefill_graph as pg


def config():
    m = json.loads(pg._DATA.read_text())["compilation"]
    c = N(**copy.deepcopy(m))
    c.mode = N(name=m["mode"])
    c.pass_config = N(**m["pass_config"])
    c.cudagraph_mode = N(name="FULL_AND_PIECEWISE")
    c.cudagraph_capture_sizes = list(pg._BASE)
    c.max_cudagraph_capture_size = 256
    return N(compilation_config=c,
        scheduler_config=N(max_num_seqs=128, max_num_batched_tokens=2048),
        parallel_config=N(tensor_parallel_size=1, pipeline_parallel_size=1,
                          decode_context_parallel_size=1, enable_dbo=False),
        speculative_config=None, offload_config=None,
        model_config=N(max_model_len=8192, dtype="torch.bfloat16", quantization="compressed-tensors",
                       hf_text_config=N(model_type="qwen3_5_text", hidden_size=2560, intermediate_size=9216,
                                        num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4, head_dim=256)),
        cache_config=N(kv_cache_memory_bytes=19*2**30, cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        additional_config={"gdn_prefill_backend": "triton"})


def test_scope_only_original_full_block_production_capacity():
    features = ("n64", "ordered", "silu", "fa2")
    with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": "production"}):
        assert pg.eligible_config(config(), features)
        for field, value in [("max_num_seqs", 32), ("max_num_batched_tokens", 1024)]:
            cfg = config(); setattr(cfg.scheduler_config, field, value)
            assert not pg.eligible_config(cfg, features)
        for features_other in [(), ("ordered", "fa2"), ("n64", "ordered", "silu")]:
            assert not pg.eligible_config(config(), features_other)
        cfg=config();cfg.model_config.hf_text_config.hidden_size=2048
        assert not pg.eligible_config(cfg, features)
        cfg=config();cfg.cache_config.mamba_ssm_cache_dtype="bfloat16"
        assert not pg.eligible_config(cfg, features)
        cfg=config();cfg.compilation_config.compile_sizes=["cudagraph_capture_sizes"]
        assert not pg.eligible_config(cfg, features)
        cfg=config();cfg.compilation_config.custom_ops=[]
        assert not pg.eligible_config(cfg, features)
        with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": "quality32"}):
            assert not pg.eligible_config(config(), features)


def test_manifest_locks_four_complete_call_and_kernel_families():
    manifest=json.loads(pg._DATA.read_text())
    assert manifest["bank_bytes"]==77578240 and len(manifest["families"])==4
    assert all(len(key)==64 and kernels and all(len(v)==64 for v in kernels.values())
               for key,kernels in manifest["families"].items())
    a=ast.parse("def f(x):\n return x+1\n").body[0]
    old=pg.ast_sha(a);a.type_params=[];assert pg.ast_sha(a)==old
    a.body[0].value.right.value=2;assert pg.ast_sha(a)!=old
    src='kernel = async_compile.triton("kernel", "def kernel(X):\\n return X + 1\\n")\nclass Runner:\n def call(self,args):\n  return args[0]+1\n'
    key,kernels=pg.aot_identity(src)
    changed=pg.aot_identity(src.replace('return X + 1','return X + 2'))
    assert key==changed[0] and kernels!=changed[1]
    assert pg.aot_identity(src.replace('args[0]+1','args[0]+2'))[0]!=key


def test_fx_rejects_hidden_module_state_and_changed_pure_kwargs():
    assert not pg.pure_fx(None)
    assert not pg.pure_fx(N(graph=N(nodes=[N(op="get_attr",target="KV_state")])))
    assert not pg.pure_fx(N(graph=N(nodes=[N(op="call_module",target="GDN")])))
    def silu(x):return x
    silu.__module__="torch.nn.functional"
    assert pg.pure_fx(N(graph=N(nodes=[N(op="call_function",target=silu,args=(1,),kwargs={})])))
    assert not pg.pure_fx(N(graph=N(nodes=[N(op="call_function",target=silu,args=(1,),kwargs={"inplace":True})])))


def test_selected_block_prefill_graph_actual_runtime():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm") is None:
        pytest.skip("requires installed 0.33 runtime")
    import torch
    import vllm.compilation.cuda_graph as cg
    import vllm.forward_context as fc
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, BatchExecutionDescriptor
    from vllm.compilation.piecewise_backend import PiecewiseBackend
    from torch._inductor.codecache import PyCodeCache
    from vllm.config.compilation import CUDAGraphMode, CompilationConfig
    from vllm_mach.fp8._prefill_inputs import ActivationBank
    cfg=config()
    actual_fields=json.loads(pg._DATA.read_text())["compilation"]
    cfg.compilation_config=CompilationConfig(**actual_fields,cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=list(pg._BASE),max_cudagraph_capture_size=256)
    worker=N(vllm_config=cfg)
    capture=N(cuda_stream=71);waits=[];main=N(cuda_stream=0,wait_stream=lambda previous:waits.append(previous.cuda_stream));active=[capture]
    key=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE,num_tokens=2048,num_reqs=32)
    ctx=N(batch_descriptor=key,cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    original_call=cg.CUDAGraphWrapper.__call__;original_route=CudaGraphManager.dispatch
    original_init=PiecewiseBackend.__init__;original_loader=PyCodeCache.__dict__["load_by_key_path"]
    marker=getattr(cg.CUDAGraphWrapper,"_mach_block_pw2048",None)
    weight=torch.ones(16,dtype=torch.bfloat16)
    def inputs(value):
        core=torch.full((2048,32,128),value,dtype=torch.bfloat16)
        gate=torch.empty_strided((2048,32,128),(12288,128,1),dtype=torch.bfloat16).fill_(value+1)
        residual=torch.full((2048,2560),value+2,dtype=torch.bfloat16)
        return core,2048,weight,gate,residual
    def backend(core,n,w,gate,residual):
        return (core.flatten()[:16]+gate.reshape(-1)[:16]+residual.flatten()[:16]+w,)
    wrapper=cg.CUDAGraphWrapper.__new__(cg.CUDAGraphWrapper)
    wrapper.runnable=backend;wrapper.vllm_config=cfg;wrapper.runtime_mode=CUDAGraphMode.PIECEWISE;wrapper.is_debugging_mode=True
    with patch.dict(pg._STATES,{},clear=True),patch.object(pg,"_HOOKS",[]), \
         patch.object(torch.cuda,"current_stream",lambda device=None:active[0]),torch.inference_mode(), \
         patch.object(fc,"get_forward_context",lambda:ctx),patch.object(fc,"is_forward_context_available",lambda:True), \
         patch.object(cg,"get_forward_context",lambda:ctx),patch.object(cg,"is_forward_context_available",lambda:True), \
         patch.object(cg,"get_offloader",lambda:N(sync_prev_onload=lambda:None)):
        try:
            if marker is not None:delattr(cg.CUDAGraphWrapper,"_mach_block_pw2048")
            assert pg.configure(worker,("n64","ordered","silu","fa2"))
            state=worker._mach_fp8_prefill_graph
            assert cfg.compilation_config.cudagraph_capture_sizes==[*pg._BASE,2048]
            state.readonly={pg.storage(weight):"weight"}
            state.islands[id(backend)]={"safe":True} # V7 math proof reused; CPU tests wrapper/copy only
            state.bank=ActivationBank('cpu');args=inputs(0);prepared,_=state.bank.prepare(args,state.readonly);fixed,_=prepared
            output=torch.empty(16,dtype=torch.bfloat16)
            entry=N(input_addresses=[t.data_ptr()for t in fixed if isinstance(t,torch.Tensor)],output=(output,))
            entry.cudagraph=N(replay=lambda:output.copy_(backend(*fixed)[0]))
            wrapper.concrete_cudagraph_entries={key:entry}
            got=wrapper(*args);assert torch.equal(got[0],backend(*args)[0])
            state.main_stream=main;active[0]=main;pg.after_compile(worker)
            assert waits==[71]and state.ready and not pg._HOOKS
            assert PiecewiseBackend.__init__ is original_init
            consumed=[]
            for value in (1,2):
                args=inputs(value);got=wrapper(*args);consumed.append(got[0].clone())
                assert torch.equal(consumed[-1],backend(*args)[0])
                assert wrapper.concrete_cudagraph_entries[key].input_addresses==entry.input_addresses
            assert not torch.equal(consumed[0],consumed[1]) # ordered consumer before original output storage reuse
            active[0]=N(cuda_stream=99);assert state.bank.prepare(args,state.readonly)[0]is None
            assert pg.inspect_worker(worker)["bank_bytes"]==77578240
            tail=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE,num_tokens=2048,num_reqs=32)
            fixed_tail=pg.exact_descriptor(tail,1992,32)
            assert fixed_tail.cg_mode==CUDAGraphMode.NONE and fixed_tail.num_tokens==1992
            assert pg.exact_descriptor(key,2048,32)is key
        finally:
            cg.CUDAGraphWrapper.__call__=original_call;CudaGraphManager.dispatch=original_route
            PiecewiseBackend.__init__=original_init;PyCodeCache.load_by_key_path=original_loader
            if marker is None:
                if hasattr(cg.CUDAGraphWrapper,"_mach_block_pw2048"):delattr(cg.CUDAGraphWrapper,"_mach_block_pw2048")
            else:cg.CUDAGraphWrapper._mach_block_pw2048=marker
    assert not torch.cuda.is_initialized()


def test_upstream_mismatch_disables_before_any_compilation_config_change(tmp_path):
    import importlib.metadata as md
    cfg=config();worker=N(vllm_config=cfg)
    before=copy.deepcopy(vars(cfg.compilation_config))
    with patch.object(md,"distribution",return_value=N(locate_file=lambda rel:tmp_path/rel)):
        assert not pg.configure(worker,("n64","ordered","silu","fa2"))
    assert vars(cfg.compilation_config)==before
    assert not hasattr(worker,"_mach_fp8_prefill_graph")
    receipt=pg.inspect_worker(worker)
    assert not receipt['enabled']and receipt['fallback'].startswith('upstream_source_mismatch:')
    unresolved=config();unresolved.compilation_config.cudagraph_capture_sizes=None
    assert not pg.configure(N(vllm_config=unresolved),('n64','ordered','silu','fa2'))
    assert unresolved.compilation_config.cudagraph_capture_sizes is None


def test_zero_safe_islands_become_ready_disabled_without_dropping_old_graphs():
    cfg=config();state=pg.State(cfg);state.wrappers={1:{'safe':False}}
    worker=N(_mach_fp8_prefill_graph=state)
    with patch.dict(pg._STATES,{id(cfg):state},clear=True),patch.object(pg,'_HOOKS',[]):
        pg.after_compile(worker)
    assert state.ready and state.fallback=='no_source_qualified_islands'
    assert not pg.inspect_worker(worker)['enabled']
    assert state.wrappers=={1:{'safe':False}} # original graph entries are not cleared


def test_selected_block_prefill_graph_warm_cache_fallback():
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('vllm') is None:
        pytest.skip('requires installed 0.33 runtime')
    import torch
    from vllm.config.compilation import CompilationConfig,CUDAGraphMode
    from vllm.compilation.piecewise_backend import PiecewiseBackend
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager,BatchExecutionDescriptor
    from torch._inductor.codecache import PyCodeCache
    cfg=config();cfg.compilation_config=CompilationConfig(**json.loads(pg._DATA.read_text())['compilation'],
        cudagraph_mode='FULL_AND_PIECEWISE',cudagraph_capture_sizes=list(pg._BASE),max_cudagraph_capture_size=256)
    worker=N(vllm_config=cfg)
    original=(PiecewiseBackend.__init__,PyCodeCache.__dict__['load_by_key_path'],CUDAGraphWrapper.__call__,CudaGraphManager.dispatch)
    marker=getattr(CUDAGraphWrapper,'_mach_block_pw2048',None)
    with patch.dict(pg._STATES,{},clear=True),patch.object(pg,'_HOOKS',[]):
        try:
            if marker is not None:delattr(CUDAGraphWrapper,'_mach_block_pw2048')
            assert pg.configure(worker,('n64','ordered','silu','fa2'))
            state=worker._mach_fp8_prefill_graph
            ranges=cfg.compilation_config.get_compile_ranges()
            cached=lambda *args:args
            backend=PiecewiseBackend(graph=None,vllm_config=cfg,piecewise_compile_index=0,total_piecewise_compiles=1,
                sym_shape_indices=[],vllm_backend=N(is_encoder=False),returns_tuple=True,
                compiled_runnables={str(r):cached for r in ranges})
            assert backend.graph is None and not state.islands[id(backend)]['safe']
            token=object();assert backend(token)==(token,) # actual load_all_ranges runnable preserved
            pg.after_compile(worker)
            assert state.ready and state.fallback=='no_source_qualified_islands'
            desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE,num_tokens=2048,num_reqs=31)
            fallback=pg.runtime_descriptor(state,desc,2048,31)
            assert fallback.cg_mode==CUDAGraphMode.NONE and fallback.num_tokens==2048 and fallback.num_reqs==31
            small=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL,num_tokens=32,num_reqs=32)
            assert pg.runtime_descriptor(state,small,32,32)is small
        finally:
            PiecewiseBackend.__init__,PyCodeCache.load_by_key_path,CUDAGraphWrapper.__call__,CudaGraphManager.dispatch=original
            if marker is None:
                if hasattr(CUDAGraphWrapper,'_mach_block_pw2048'):delattr(CUDAGraphWrapper,'_mach_block_pw2048')
            else:CUDAGraphWrapper._mach_block_pw2048=marker
    assert not torch.cuda.is_initialized()
