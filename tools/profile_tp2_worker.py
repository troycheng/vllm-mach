"""Diagnostic-only graph event probes. Imported by vLLM workers before load."""
import functools
import json
import os

import torch

DETAIL = os.environ.get('MACH_TP2_PROFILE') == '1'
PAIRS = {}
INVENTORY = []
SAMPLES = []
CURRENT_M = None
ACTIVE = False
STEP = 0
PROFILER = None
TRACE_DIR = None
TARGET_ROWS = None
BATCHES = []


def wrap(obj, attr, name, root=False):
    original = getattr(obj, attr)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        global CURRENT_M
        previous = CURRENT_M
        if root:
            tokens = kwargs.get('input_ids', args[0] if args else None)
            CURRENT_M = int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else None
        enabled = DETAIL and torch.cuda.is_current_stream_capturing() and CURRENT_M in (1, 2, 4, 8, 16, 24, 32)
        if enabled:
            key = (CURRENT_M, name(*args, **kwargs) if callable(name) else name)
            if key not in PAIRS:
                PAIRS[key] = tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            start, end = PAIRS[key]
            start.record()
        try:
            return original(*args, **kwargs)
        finally:
            if enabled:
                end.record()
            if root:
                CURRENT_M = previous
    setattr(obj, attr, measured)


def instrument(model):
    # Install shape/M overrides before workspace planning and graph capture.
    # Each candidate still rotates the entire checkpoint through real TP2 calls.
    overrides=json.loads(os.environ.get('MACH_TP2_GEMM_OVERRIDES','[]'))
    if overrides:
        import mxfp6
        mxfp6.load_library()
        weights={(int(layer.weight.shape[0]),int(layer.weight.shape[1])*4//3):layer.weight
                 for layer in model.modules()
                 if type(getattr(getattr(layer,'scheme',None),'ocp_mx_linear',None)).__name__=='Mxfp6Sm120LinearKernel'}
        for candidate in overrides:
            m,n,k,config,swizzle,raster=candidate
            assert m in (1,2,4,8,16,24,32) and (n,k) in weights
            torch.ops.mxfp6.set_w6a8_config(weights[n,k],m,n,k,config,swizzle,raster,torch.bfloat16)
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        kernel = getattr(getattr(module, 'scheme', None), 'ocp_mx_linear', None)
        INVENTORY.append(dict(name=name, cls=type(module).__name__, kernel=type(kernel).__name__,
                              weight_shape=list(weight.shape) if isinstance(weight, torch.Tensor) else None,
                              fused_ar_norm=getattr(module, 'use_fused_ar_gemma_norm', None)))
        if name and name.endswith(('linear_attn', 'self_attn', 'mlp', 'gate_up_proj', 'down_proj',
                                   'act_fn', 'in_proj_qkvz', 'in_proj_ba', 'out_proj', 'o_proj', 'qkv_proj', 'norm')):
            wrap(module, 'forward', name)
        if hasattr(module, '_rms_norm_gated_cuda'):
            wrap(module, '_rms_norm_gated_cuda', name + '.gated_norm')
    names = {id(module): name for name, module in model.named_modules()}
    from vllm.model_executor.models import qwen3_5
    wrap(qwen3_5, 'fused_allreduce_gemma_rms_norm',
         lambda x, residual, norm, **kw: names[id(norm)] + '.fused_ar')
    wrap(model, 'forward', 'model', root=True)


class TP2Probe:
    def tp2_begin(self, trace_dir, target_rows):
        global ACTIVE, STEP, TRACE_DIR, TARGET_ROWS
        ACTIVE, STEP, TRACE_DIR = True, 0, trace_dir
        TARGET_ROWS = target_rows
        BATCHES.clear()
        SAMPLES.clear()
        torch.cuda.reset_peak_memory_stats()

    def tp2_end(self):
        global ACTIVE, PROFILER
        ACTIVE = False
        torch.cuda.synchronize()
        if PROFILER is not None:
            PROFILER.stop()
            PROFILER.export_chrome_trace(f'{TRACE_DIR}/rank{self.rank}.json')
            PROFILER = None
        from vllm_mach.mxfp6.gdn_decode import stats
        return dict(rank=self.rank, samples=SAMPLES, batches=BATCHES, inventory=INVENTORY, gdn=stats(),
                    fused_swiglu_layers=sum(bool(getattr(m, "_mach_swiglu_prepared", False))
                                            for m in self.model_runner.model.modules()),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())


from vllm.v1.worker.gpu.model_runner import GPUModelRunner
_load = GPUModelRunner.load_model

def load(self, *args, **kwargs):
    result = _load(self, *args, **kwargs)
    instrument(self.model)
    return result
GPUModelRunner.load_model = load
_execute = GPUModelRunner.execute_model
_sample = GPUModelRunner.sample_tokens
PENDING = None


def execute(self, scheduler_output, *args, **kwargs):
    global STEP, PROFILER, PENDING
    counts = scheduler_output.num_scheduled_tokens
    decode = bool(counts) and all(v == 1 for v in counts.values()) and not scheduler_output.scheduled_new_reqs
    PENDING = None
    if ACTIVE and DETAIL and decode and len(counts) == TARGET_ROWS:
        STEP += 1
        if STEP == 40:
            PROFILER = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            PROFILER.start()
        if 40 <= STEP <= 42:
            PENDING = len(counts)
        if STEP == 43 and PROFILER is not None:
            PROFILER.stop()
            from vllm.distributed import get_tensor_model_parallel_rank
            PROFILER.export_chrome_trace(f'{TRACE_DIR}/rank{get_tensor_model_parallel_rank()}.json')
            PROFILER = None
    return _execute(self, scheduler_output, *args, **kwargs)


def sample(self, *args, **kwargs):
    batch = self.execute_model_state.input_batch if self.execute_model_state is not None else None
    physical_rows = int(batch.num_tokens_after_padding) if batch is not None else None
    if ACTIVE and batch is not None:
        BATCHES.append(dict(logical_rows=int(batch.num_reqs), padded_rows=physical_rows,
                            has_prefill=bool(batch.has_prefill)))
    result = _sample(self, *args, **kwargs)
    if PENDING is not None:
        torch.cuda.synchronize()
        m = physical_rows
        assert m in (1, 2, 4, 8, 16, 24, 32)
        timings = {name: start.elapsed_time(end) for (rows, name), (start, end) in PAIRS.items() if rows == m}
        SAMPLES.append(dict(step=STEP, logical_rows=PENDING, padded_rows=m, inclusive_ms=timings))
    return result
GPUModelRunner.execute_model = execute
GPUModelRunner.sample_tokens = sample
