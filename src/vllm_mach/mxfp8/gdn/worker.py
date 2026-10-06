# SPDX-License-Identifier: Apache-2.0
"""Complete ordered W4 state lifecycle inside vLLM's existing opaque GDN op.

Imports and hook registration are CPU only. Kernels load after Worker.init_device.
"""
import contextvars
import functools
import hashlib
import importlib
import inspect
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GDN_SHA = "8ea18f04ee77359e8dea3e089bb991219468f7985648dc4e24eacb9cd20cf719"
RUNNER_SHA = "7e6de891206992f6018bb2b26a2c2fff5f3eabe3de1fc9066f5eb8f7e62fb52d"
FLA_SHA = "fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755"
W = 4
ROWS = (1,2,4,8,16)+tuple(range(24,129,8))
_ELIGIBLE_ROWS = ()
_MAX_NUM_SEQS = None
_RUN_MODE = None
_QUALITY_ROW = None
_PREPARATION_BYTES = {}
_COMPONENT = None
_ACTIVE_TARGET = contextvars.ContextVar("mach_mxfp8_ordered_target",default=None)
_ORIG_CORE = None
_ORIG_DECODE = None
_READY = False
_OWNER = None
_CACHE_INITIALIZED = False
_LAYERS = {}
_VIEW_SCRATCH = {}
_CAPTURE = {}
_EAGER = {}
_FALLBACK = {}
_PROFILE = {}
_ALLOCATION_RESETS = {"new":0,"cached":0,"ids":0}


def _component():
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = importlib.import_module(".ordered_full_m4_m16_deferred_dispatch", __package__)
    return _COMPONENT


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _view_key(state):
    return (state.data_ptr(), tuple(state.shape), tuple(state.stride()),
            str(state.device), str(state.dtype))

def _pools(layer):
    return layer.kv_cache[0],layer.kv_cache[1]

def _mode_contract():
    mode = os.environ.get("VLLM_MACH_MXFP8_MODE", "production")
    if mode == "production":
        if os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS") is not None:
            raise RuntimeError("quality rows require explicit quality mode")
        return mode, None, 19*2**30
    if mode == "quality":
        row = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
        if row not in ("32", "64"):
            raise RuntimeError("quality mode requires fixed rows32 or64")
        return mode, int(row), (4 if row == "32" else 19)*2**30
    raise RuntimeError("unknown MXFP8 mode; select production or quality")


def _check_config(worker):
    import torch
    cfg = worker.vllm_config
    cache,model = cfg.cache_config,cfg.model_config
    mode, row, budget = _mode_contract()
    if _RUN_MODE is not None and (mode, row) != (_RUN_MODE, _QUALITY_ROW):
        raise RuntimeError("GDN profile mode changed after worker initialization")
    if (cfg.parallel_config.tensor_parallel_size != 1
            or cfg.speculative_config is not None
            or cache.enable_prefix_caching
            or cache.mamba_cache_mode != "none"
            or cache.use_replayssm
            or getattr(cache,"kv_offloading_size",None) not in (None,0)
            or cache.cache_dtype not in ("auto","bfloat16","fp8_e4m3")
            or model.dtype != torch.bfloat16
            or getattr(cfg,"kv_transfer_config",None) is not None):
        raise RuntimeError("deferred GDN requires TP1, no MTP/prefix/ReplaySSM/connector, BF16 model, BF16 or calibrated FP8 KV, mamba none")
    maximum = row if mode == "quality" else 128
    length = 1024 if mode == "quality" else 8192
    if (cache.kv_cache_memory_bytes != budget
            or model.max_model_len != length
            or cfg.scheduler_config.max_num_seqs != maximum):
        raise RuntimeError(f"ordered GDN requires maxseq{maximum}/maxlen{length}/{budget//2**30} GiB KV in {mode} mode")
    if mode == "quality" and getattr(cfg.scheduler_config, "max_num_batched_tokens", None) != row*256:
        raise RuntimeError(f"quality rows{row} requires maxbatch{row*256}")
    if getattr(model,"cpu_offload_gb",0) not in (0,None):
        raise RuntimeError("CPU weight offload unsupported")

def _layout(layer,state):
    import torch
    return (layer.tp_size == 1 and not layer.gqa_interleaved_layout
            and layer.enable_packed_recurrent_decode and layer._is_sm120
            and layer.num_k_heads == 16 and layer.num_v_heads == 32
            and layer.head_k_dim == layer.head_v_dim == 128
            and len(layer.kv_cache) == 2
            and state.dtype == torch.float32
            and tuple(state.shape[1:]) == (32,128,128)
            and tuple(state.stride()[1:]) == (16384,128,1)
            and state.stride(0) >= 524288
            and layer.kv_cache[0].dtype == torch.bfloat16
            and layer.A_log.dtype == torch.float32
            and layer.dt_bias.dtype == torch.bfloat16)

def _metadata(layer):
    from vllm.forward_context import get_forward_context
    raw = get_forward_context().attn_metadata
    return raw.get(layer.prefix) if isinstance(raw,dict) else None

def _target(layer,md,qkv,b,a,state):
    import torch
    m = int(qkv.shape[0])
    ids = md.non_spec_state_indices_tensor if md is not None else None
    return (_READY and layer.prefix in _LAYERS and m in _ELIGIBLE_ROWS
            and md is not None and md.spec_sequence_masks is None
            and md.num_spec_decodes == 0 and md.num_prefills == 0
            and md.num_decodes > 0 and md.num_actual_tokens == m
            and ids is not None and ids.numel() >= m
            and ids.dtype in (torch.int32,torch.int64)
            and ids.device == state.device
            and qkv.dtype == b.dtype == a.dtype == torch.bfloat16
            and tuple(qkv.shape) == (m,8192)
            and tuple(b.shape) == tuple(a.shape) == (m,32)
            and qkv.stride(1) == b.stride(1) == a.stride(1) == 1)

def _relevant(md):
    # In this frozen no-MTP configuration the builder's non-spec vector is
    # block_table_tensor[:,0], one slot per request; 0 padding may repeat.
    if md.spec_sequence_masks is not None or md.num_spec_decodes:
        raise RuntimeError("spec/MTP route violates deferred contract")
    ids = md.non_spec_state_indices_tensor
    if ids is None:
        if md.num_prefills or md.num_decodes:
            raise RuntimeError("stock route has no non-spec state indices")
        return None
    active_reqs = md.num_decodes + md.num_prefills
    if active_reqs < 0 or ids.numel() < active_reqs:
        raise RuntimeError("non-spec index buffer shorter than active requests")
    if active_reqs == 0:
        return None
    return ids[:active_reqs]

def _register_gdn():
    global _ORIG_CORE,_ORIG_DECODE
    import torch
    component = _component()
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    if getattr(GDN,"_mach_mxfp8_ordered_installed",False): return
    if getattr(GDN,"_mx8_deferred_installed",False):
        raise RuntimeError("old deferred hook conflicts with full-M hook")
    if any(getattr(GDN,name,False) for name in
           ("_mx8_ssm_full_installed","_ssm_lowm_capture",
            "_gdn_packed_capture","_mx8_full_shadow_installed")):
        raise RuntimeError("conflicting GDN hook is installed")
    if _sha(inspect.getsourcefile(GDN)) != GDN_SHA:
        raise RuntimeError("live GDN source SHA drift")
    fla_source = importlib.import_module(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent")
    if _sha(fla_source.__file__) != FLA_SHA:
        raise RuntimeError("live packed FLA source SHA drift")
    _ORIG_CORE = GDN._forward_core
    _ORIG_DECODE = GDN._forward_core_decode_non_spec

    @functools.wraps(_ORIG_CORE)
    def core(self,mixed_qkv,b,a,core_attn_out,hidden_states=None):
        md = _metadata(self)
        key = (self.prefix,int(mixed_qkv.shape[0]))
        if md is None:
            _PROFILE[key] = _PROFILE.get(key,0)+1
            return _ORIG_CORE(self,mixed_qkv,b,a,core_attn_out,hidden_states)
        if not _READY:
            raise RuntimeError("GDN metadata reached before deferred pools are prepared")
        _,state = _pools(self)
        if not _layout(self,state):
            raise RuntimeError(f"GDN live pool layout changed: {self.prefix}")
        if _target(self,md,mixed_qkv,b,a,state):
            capturing = torch.cuda.is_current_stream_capturing()
            counter = _CAPTURE if capturing else _EAGER
            counter[key] = counter.get(key,0)+1
            token = _ACTIVE_TARGET.set(self)
            try:
                # Only this eligible path suppresses the FI probe; the captured
                # real parent took packed FLA and returned false from FI.
                return _ORIG_CORE(self,mixed_qkv,b,a,core_attn_out,
                                  hidden_states=None)
            finally:
                _ACTIVE_TARGET.reset(token)
        _FALLBACK[key] = _FALLBACK.get(key,0)+1
        ids = _relevant(md)
        if ids is not None:
            scratch = _VIEW_SCRATCH[_view_key(state)]
            component.materialize_slots(state,ids,scratch)
        # materialize_slots already resets age=0/prefix=1 before stock reads
        # base. Stock updates only base, so a second metadata launch is redundant.
        return _ORIG_CORE(self,mixed_qkv,b,a,core_attn_out,hidden_states)

    @functools.wraps(_ORIG_DECODE)
    def decode(self,mixed_qkv,b,a,core_attn_out,attn_metadata):
        if _ACTIVE_TARGET.get() is not self:
            return _ORIG_DECODE(self,mixed_qkv,b,a,core_attn_out,attn_metadata)
        m = int(attn_metadata.num_actual_tokens)
        ids = attn_metadata.non_spec_state_indices_tensor[:m]
        conv = self.kv_cache[0]
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1,-2)
        state = self.kv_cache[1]
        # Exact original convolution from _forward_core_decode_non_spec.
        weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                          self.conv1d.weight.size(2))
        post = gdn.causal_conv1d_update(
            mixed_qkv[:m],conv,weights,self.conv1d.bias,self.activation,
            conv_state_indices=ids,validate_data=False)
        parent_out = core_attn_out[:m].unsqueeze(1)
        live = _VIEW_SCRATCH[_view_key(state)]
        component.decode(post,a[:m],b[:m],self.A_log,self.dt_bias,
                         state,parent_out,ids,live,scale=128**-0.5)
        return

    GDN._forward_core = core
    GDN._forward_core_decode_non_spec = decode
    GDN._mach_mxfp8_ordered_installed = True

def _prepare_layers(worker):
    _claim_worker(worker)
    global _ELIGIBLE_ROWS, _MAX_NUM_SEQS
    import torch
    from .. import native_backend
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    _check_config(worker)
    _component()
    maximum = int(worker.vllm_config.scheduler_config.max_num_seqs)
    if maximum <= 0 or _MAX_NUM_SEQS is not None:
        raise RuntimeError("invalid/repeated full-M row preparation")
    _MAX_NUM_SEQS = maximum
    # Accepted short-quality runs used the same all-row set, clipped by
    # scheduler capacity; captures 4/32/64 and their tails stay ordered.
    _ELIGIBLE_ROWS = tuple(m for m in ROWS if m <= maximum)
    native = native_backend.inspect(worker)
    if not native["installed"] or native["policy"] != "all":
        raise RuntimeError("MXFP8 native(all) required")
    layers = [x for _,x in worker.get_model().named_modules() if isinstance(x,GDN)]
    if len(layers) != 24 or len({x.prefix for x in layers}) != 24:
        raise RuntimeError("expected 24 unique GDN layers")
    for i,layer in enumerate(layers):
        _LAYERS[layer.prefix] = i
        if (layer.tp_size != 1 or layer.gqa_interleaved_layout
                or layer.num_k_heads != 16 or layer.num_v_heads != 32
                or layer.head_k_dim != 128 or layer.head_v_dim != 128
                or not layer.enable_packed_recurrent_decode
                or layer.A_log.dtype != torch.float32
                or layer.dt_bias.dtype != torch.bfloat16):
            raise RuntimeError(f"unsupported layer {layer.prefix}")

def _prepare_pools(worker):
    _claim_worker(worker)
    if not _CACHE_INITIALIZED:
        raise RuntimeError("initialize ordered GDN cache before pool warmup/capture")
    global _READY
    import torch
    component = _component()
    from . import metadata_reset as kernels
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
    if _READY or len(_LAYERS) != 24:
        raise RuntimeError("bad deferred preparation lifecycle")
    if worker.model_runner.kv_connector is not NO_OP_KV_CONNECTOR:
        raise RuntimeError("KV connector/offload path unsupported")
    layers = [x for _,x in worker.get_model().named_modules() if isinstance(x,GDN)]
    views = {}
    for layer in layers:
        _,state = _pools(layer)
        if not _layout(layer,state):
            raise RuntimeError(f"unsupported bound pool {layer.prefix}")
        views.setdefault(_view_key(state),state)
    if len(views) != 8:
        raise RuntimeError(f"expected 8 unique GDN state views, got {len(views)}")
    if len({tuple(state.stride()) for state in views.values()}) != 1:
        raise RuntimeError("GDN state views have different page strides")
    required = 0
    for state in views.values():
        for name,shape in component.required_shapes(state.shape[0],W).items():
            required += __import__('math').prod(shape) * (4 if name=='age' else 4)
    _PREPARATION_BYTES.update(live_scratch=required,reserve=512*2**20)
    free,total = torch.cuda.mem_get_info(layers[0].A_log.device)
    if free < required + 512*2**20:
        raise RuntimeError(f"ordered GDN requires {required} B live scratch + 512 MiB reserve; GPU free {free}/{total}")
    for key,state in views.items():
        shape = component.required_shapes(state.shape[0],W)
        _VIEW_SCRATCH[key] = component.DeferredScratch(**{
            name:torch.zeros(size,dtype=(torch.int32 if name=="age" else torch.float32),
                             device=state.device) for name,size in shape.items()})
        # Initialize only metadata once. Real new assignments are reset again
        # by the scheduler hooks; this protects positive dummy graph captures.
        all_slots = torch.arange(1,state.shape[0],dtype=torch.int32,
                                 device=state.device)
        kernels.reset_metadata(all_slots,_VIEW_SCRATCH[key],state.shape[0])
    device = layers[0].A_log.device
    # Warm every call shape before CUDA graph capture without writing live
    # base state. Zero indices exercise the invalid-slot guards.
    for layer in layers:
        _,state = _pools(layer)
        ids = torch.zeros((max(_ELIGIBLE_ROWS,default=1),),dtype=torch.int32,device=state.device)
        scratch = _VIEW_SCRATCH[_view_key(state)]
        kernels.reset_metadata(ids,scratch,state.shape[0])
        # Fallback materialization also needs warming when no target M fits.
        component.materialize_slots(state,ids,scratch)
        for m in _ELIGIBLE_ROWS:
            idx = ids[:m]
            # The real split QKV and post-conv tensor have row stride 12288,
            # which is a Triton constexpr. Warm that exact graph signature.
            qkv = torch.empty_strided((m,8192),(12288,1),
                                      dtype=torch.bfloat16,device=state.device)
            qkv.zero_()
            ba = torch.zeros((m,32),dtype=torch.bfloat16,device=state.device)
            out = torch.empty((m,1,32,128),dtype=torch.bfloat16,device=state.device)
            component.decode(qkv,ba,ba,layer.A_log,layer.dt_bias,
                             state,out,idx,scratch,scale=128**-0.5)
            component.materialize_slots(state,idx,scratch)
    torch.cuda.synchronize(device)
    _READY = True

def _flatten_new(blocks):
    if blocks is None: return []
    if not isinstance(blocks,tuple):
        raise TypeError("block_ids must be tuple[list[int], ...]")
    ids = []
    for group in blocks:
        if not isinstance(group,list) or any(type(x) is not int for x in group):
            raise TypeError("block_ids group must contain Python ints")
        ids.extend(x for x in group if x > 0)
    return ids

def _reset_allocated(ids,kind):
    if not ids or not _READY: return
    import torch
    from . import metadata_reset as kernels
    unique = sorted(set(ids))
    # CPU scheduler knows these newly assigned IDs. No device age read or
    # capture-time host slot value is involved.
    ids_by_device = {}
    for key,scratch in _VIEW_SCRATCH.items():
        pages = key[1][0]
        if unique[-1] >= pages:
            raise RuntimeError(f"allocated block outside GDN state pool: {unique[-1]}/{pages}")
        dev = torch.device(key[3])
        if key[3] not in ids_by_device:
            ids_by_device[key[3]] = torch.tensor(unique,dtype=torch.int32,device=dev)
        device_ids = ids_by_device[key[3]]
        kernels.reset_metadata(device_ids,scratch,pages)
    _ALLOCATION_RESETS[kind] += 1
    _ALLOCATION_RESETS["ids"] += len(unique)

def _install_runner_lifecycle():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    if getattr(GPUModelRunner,"_mach_mxfp8_ordered_lifecycle",False): return
    if getattr(GPUModelRunner,"_mx8_deferred_lifecycle",False):
        raise RuntimeError("old deferred runner lifecycle conflicts with full-M hook")
    if _sha(inspect.getsourcefile(GPUModelRunner)) != RUNNER_SHA:
        raise RuntimeError("GPUModelRunner lifecycle source SHA drift")
    old_add,old_update = GPUModelRunner.add_requests,GPUModelRunner.update_requests
    @functools.wraps(old_add)
    def add(self,scheduler_output):
        if any(req.num_computed_tokens != 0
               for req in scheduler_output.scheduled_new_reqs):
            raise RuntimeError("new/resumed request with retained state requires an explicit deferred-state transfer")
        result = old_add(self,scheduler_output)
        ids = [x for req in scheduler_output.scheduled_new_reqs
               for x in _flatten_new(req.block_ids)]
        _reset_allocated(ids,"new")
        return result
    @functools.wraps(old_update)
    def update(self,scheduler_output):
        if scheduler_output.kv_cache_block_copies:
            raise RuntimeError("deferred GDN does not support block copies")
        result = old_update(self,scheduler_output)
        ids = [x for block in scheduler_output.scheduled_cached_reqs.new_block_ids
               for x in _flatten_new(block)]
        _reset_allocated(ids,"cached")
        return result
    GPUModelRunner.add_requests = add
    GPUModelRunner.update_requests = update
    GPUModelRunner._mach_mxfp8_ordered_lifecycle = True

def inspect_worker(worker):
    _claim_worker(worker)
    if _READY:
        _bound_views(worker)
    component = _component()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    layers = [x for _,x in worker.get_model().named_modules() if isinstance(x,GDN)]
    sources = {path.name:_sha(path) for path in ROOT.glob("*.py")}
    mode, quality_row, budget = _mode_contract()
    record = {"mode":"quality" if mode == "quality" else "normal",
        "profile_mode":mode,"quality_rows":quality_row,
        "kv_cache_memory_bytes":budget,
        "production_throughput_profile":mode == "production",
        "implementation":"ordered_full_w4",
        "configured_rows":list(ROWS),
        "eligible_rows":list(_ELIGIBLE_ROWS),
        "scheduler_max_num_seqs":_MAX_NUM_SEQS,
        "warm_index_capacity":max(_ELIGIBLE_ROWS,default=1),
        "preparation_required_bytes":dict(_PREPARATION_BYTES),
        "component_module":component.__name__,
        "component_sha256":_sha(component.__file__),
        "coeff_semantics":"raw_alpha_ordered_replay_M4_M16_deferred",
        "ready":_READY,"cache_initialized":_CACHE_INITIALIZED,
        "layers":sorted(_LAYERS),"layer_count":len(_LAYERS),
        "unique_state_views":len(_VIEW_SCRATCH),
        "view_pages_strides":[{"pages":key[1][0],"stride":key[2],
                              "device":key[3]} for key in _VIEW_SCRATCH],
        "capture_construction":{f"{p}|m{m}":n for (p,m),n in _CAPTURE.items()},
        "eager_dispatch":{f"{p}|m{m}":n for (p,m),n in _EAGER.items()},
        "fallback_dispatch":{f"{p}|m{m}":n for (p,m),n in _FALLBACK.items()},
        "profile_dispatch":{f"{p}|m{m}":n for (p,m),n in _PROFILE.items()},
        "new_block_metadata_resets":dict(_ALLOCATION_RESETS),
        "scratch_bytes":sum(sum(getattr(s,n).numel()*getattr(s,n).element_size()
                  for n in ("pending_k","pending_d","coeff","prefix","age"))
                  for s in _VIEW_SCRATCH.values()),
        "sources":sources,
        "fla_sha256":FLA_SHA,"gdn_sha256":GDN_SHA,"runner_sha256":RUNNER_SHA,
        "normal_gpu_counter_allocated":False,
        "existing_outer_custom_op_only":True,
        "bound_pool_shapes":[list(x.kv_cache[1].shape) for x in layers]}
    return record

def install_hook():
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker,"_mach_mxfp8_ordered_worker_hook",False): return False
    if (getattr(Worker,"_mx8_deferred_worker_hook",False) or
            getattr(Worker,"_mx8_deferred_allm_worker_hook",False)):
        raise RuntimeError("existing deferred Worker hook conflicts with full-M hook")
    old_init,old_load,old_compile = Worker.init_device,Worker.load_model,Worker.compile_or_warm_up_model
    old_cache = Worker.initialize_from_config
    @functools.wraps(old_init)
    def init(self,*args,**kwargs):
        result=old_init(self,*args,**kwargs)
        install_runtime(self)
        return result
    @functools.wraps(old_load)
    def load(self,*args,**kwargs):
        result=old_load(self,*args,**kwargs)
        _prepare_layers(self)
        return result
    @functools.wraps(old_cache)
    def initialize_from_config(self,*args,**kwargs):
        if _READY or _CACHE_INITIALIZED:
            raise RuntimeError("ordered GDN does not support cache reallocation in a live worker")
        result = old_cache(self,*args,**kwargs)
        initialize_cache(self)
        return result
    @functools.wraps(old_compile)
    def compile_model(self,*args,**kwargs):
        _prepare_pools(self)
        result=old_compile(self,*args,**kwargs)
        inspect_worker(self)
        return result
    Worker.init_device,Worker.load_model,Worker.compile_or_warm_up_model = init,load,compile_model
    Worker.initialize_from_config = initialize_from_config
    Worker._mach_mxfp8_ordered_worker_hook = True
    return True


def _claim_worker(worker):
    global _OWNER
    if _OWNER is None:
        _OWNER = worker  # Retain the owner, not an id that can be recycled.
    elif _OWNER is not worker:
        raise RuntimeError("ordered GDN state already belongs to another worker")


def install_runtime(worker):
    """After init_device: register the opaque-op replacement and scheduler resets."""
    global _RUN_MODE, _QUALITY_ROW
    _claim_worker(worker)
    _check_config(worker)
    _RUN_MODE, _QUALITY_ROW, _ = _mode_contract()
    _register_gdn()
    _install_runner_lifecycle()


def initialize_cache(worker):
    """After initialize_from_config: validate state binding without allocating scratch."""
    global _CACHE_INITIALIZED
    _claim_worker(worker)
    if _READY or _CACHE_INITIALIZED or len(_LAYERS) != 24:
        raise RuntimeError("ordered GDN requires model load, then a single cache binding")
    _check_config(worker)
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    for _, layer in worker.get_model().named_modules():
        if isinstance(layer, GDN) and not _layout(layer, _pools(layer)[1]):
            raise RuntimeError(f"unsupported initialized GDN state: {layer.prefix}")
    _CACHE_INITIALIZED = True


def _idle_boundary(worker):
    import torch
    _claim_worker(worker)
    if not _READY or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("state lifecycle operation requires ready pools outside graph capture")
    if worker.model_runner.req_states.req_id_to_index:
        raise RuntimeError("drain requests before explicit state lifecycle operations")
    torch.cuda.synchronize(worker.device)


def materialize_state(worker):
    """Drain-boundary RPC: flush all positive slots before exporting dense FP32 state."""
    import torch
    _idle_boundary(worker)
    component = _component()
    views = _bound_views(worker)
    for key, scratch in _VIEW_SCRATCH.items():
        state = views[key]
        ids = torch.arange(1, state.shape[0], dtype=torch.int32, device=state.device)
        component.materialize_slots(state, ids, scratch)
    torch.cuda.synchronize(worker.device)
    return {"materialized_views": len(_VIEW_SCRATCH), "synchronized": True}


def reset_state(worker, *, zero_base=False):
    """Drain-boundary RPC: invalidate pending terms, optionally clear dense state.

    Scratch/owner storage stays stable for previously captured graphs. Normal new
    request and slot reuse resets happen automatically in the scheduler hooks.
    """
    import torch
    if type(zero_base) is not bool:
        raise TypeError("zero_base must be bool")
    _idle_boundary(worker)
    component = _component()
    views = _bound_views(worker)
    for key, scratch in _VIEW_SCRATCH.items():
        state = views[key]
        ids = torch.arange(1, state.shape[0], dtype=torch.int32, device=state.device)
        component.reset_slots(state, ids, scratch, zero_base=zero_base)
    torch.cuda.synchronize(worker.device)
    return {"reset_views": len(_VIEW_SCRATCH), "zero_base": zero_base,
            "scratch_storage_preserved": True, "synchronized": True}


def _bound_views(worker):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    views = {_view_key(layer.kv_cache[1]): layer.kv_cache[1]
             for _, layer in worker.get_model().named_modules() if isinstance(layer, GDN)}
    if set(views) != set(_VIEW_SCRATCH):
        raise RuntimeError("bound GDN state pool storage changed after capture")
    return views


# Direct lifecycle entrypoints for the complete profile's composed Worker hook.
prepare_layers = _prepare_layers
prepare_pools = _prepare_pools
install_worker_hook = install_hook
