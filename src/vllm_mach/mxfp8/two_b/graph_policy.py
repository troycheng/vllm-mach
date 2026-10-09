"""Frozen FULL decode / exact2048 PIECEWISE graph policy for the 2B profile."""
from functools import wraps
import os

BASE_CAPTURE_SIZES = (4, 8, 16, 24, 32, 48, 64, 96, 128, 160)
CAPTURE_SIZES = (*BASE_CAPTURE_SIZES, 2048)
_INSTALLED = False


def _contract():
    mode = os.environ.get("VLLM_MACH_MXFP8_MODE", "production")
    row = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
    if mode == "production" and row is None:
        return mode, BASE_CAPTURE_SIZES, CAPTURE_SIZES, 160, 16384, 2048, 19 * 2**30
    if mode == "quality" and row in ("4", "32", "64"):
        rows = int(row)
        captures = tuple(m for m in (4, 32, 64) if m <= rows)
        return mode, captures, captures, rows, 1024, max(8192, rows * 256), 4 * 2**30
    raise RuntimeError("2B graphs require production or explicit quality rows4/32/64")


def compilation_settings():
    _, _, captures, *_ = _contract()
    return {"cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(captures),
            "max_cudagraph_capture_size": captures[-1]}


def apply_descriptor(desc, num_tokens, num_reqs):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    if num_tokens > 160 and num_tokens != 2048:
        # Before prepare_inputs: large tails keep their actual shape.
        return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens, num_reqs=num_reqs,
            num_active_loras=desc.num_active_loras)
    if num_tokens == 2048 and (desc.cg_mode != CUDAGraphMode.PIECEWISE
                              or desc.num_tokens != 2048):
        raise RuntimeError("actual2048 requires captured exact PIECEWISE2048")
    return desc


def verify_capture(worker):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
    mode, full, captures, maxseq, maxlen, maxbatch, capacity = _contract()
    cfg, runner = worker.vllm_config, worker.model_runner
    manager = runner.cudagraph_manager
    parallel = cfg.parallel_config
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or cfg.lora_config is not None
            or cfg.speculative_config is not None
            or runner.kv_connector is not NO_OP_KV_CONNECTOR
            or cfg.model_config.max_model_len != maxlen
            or cfg.scheduler_config.max_num_seqs != maxseq
            or cfg.scheduler_config.max_num_batched_tokens != maxbatch
            or cfg.cache_config.kv_cache_memory_bytes != capacity
            or cfg.cache_config.enable_prefix_caching
            or manager is None or manager.use_breakable_cg
            or not manager._graphs_captured):
        raise RuntimeError("2B graph startup capacity/runtime contract changed")
    comp = cfg.compilation_config
    planned = {key.name: sorted(d.num_tokens for d in descs)
               for key, descs in manager._capture_descs.items() if descs}
    expected = {"PIECEWISE": list(captures), "FULL": list(full)}
    if (comp.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
            or manager.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
            or comp.max_cudagraph_capture_size != captures[-1]
            or comp.cudagraph_capture_sizes != list(captures)
            or planned != expected):
        raise RuntimeError(f"2B graph capture plan differs: {planned}")
    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    if (set(manager.graphs) != set(full_descs)
            or any(graph is None for graph in manager.graphs.values())):
        raise RuntimeError("2B FULL graph capture incomplete")
    pw_counts = {}
    for wrapper in list(CUDAGraphWrapper._all_instances):
        if (wrapper.vllm_config is not cfg
                or wrapper.runtime_mode != CUDAGraphMode.PIECEWISE):
            continue
        for desc, entry in wrapper.concrete_cudagraph_entries.items():
            if entry.cudagraph is None:
                raise RuntimeError("2B PIECEWISE graph capture incomplete")
            pw_counts[desc.num_tokens] = pw_counts.get(desc.num_tokens, 0) + 1
    if sorted(pw_counts) != list(captures):
        raise RuntimeError(f"2B PIECEWISE graph capture incomplete: {pw_counts}")
    # Do not arm dispatch until real graphs and the fixed capacity are verified.
    manager._mach_2b_graph_mode = mode
    for tokens in range(1, maxseq + 1):
        padded = next(m for m in full if m >= tokens)
        for uniform, route in ((1, CUDAGraphMode.FULL), (None, CUDAGraphMode.PIECEWISE)):
            desc = manager.dispatch(num_reqs=tokens if uniform else 1,
                                    num_tokens=tokens, uniform_token_count=uniform,
                                    num_active_loras=0, max_query_len=1 if uniform else tokens)
            if desc.cg_mode != route or desc.num_tokens != padded:
                raise RuntimeError(f"2B graph dispatch changed at {tokens} tokens")
    probes = (161, 2047, 2049) if mode == "production" else (maxseq + 1,)
    for tokens in probes:
        desc = manager.dispatch(num_reqs=1, num_tokens=tokens,
                                uniform_token_count=None, num_active_loras=0)
        if desc.cg_mode != CUDAGraphMode.NONE or desc.num_tokens != tokens:
            raise RuntimeError("2B uncaptured tails require NONE without padding")
    if mode == "production":
        desc = manager.dispatch(num_reqs=1, num_tokens=2048,
                                uniform_token_count=None, num_active_loras=0)
        if desc.cg_mode != CUDAGraphMode.PIECEWISE or desc.num_tokens != 2048:
            raise RuntimeError("actual2048 requires captured exact PIECEWISE2048")
    return {"ready": True, "mode": mode, "captures": planned,
            "capture_sizes": list(captures), "full_sizes": list(full),
            "piecewise_sizes": list(captures),
            "piecewise_graph_counts": {str(m): n for m, n in sorted(pw_counts.items())},
            "kv_capacity_bytes": capacity, "large_tail_mode": "NONE",
            "large_tail_padding": False}


def install_dispatch():
    """Install only manager dispatch; worker lifecycle belongs to the profile."""
    global _INSTALLED
    if os.environ.get("VLLM_MACH_PROFILE") != "qwen35-2b-mxfp8-champion-v1":
        raise RuntimeError("2B graph policy requires qwen35-2b-mxfp8-champion-v1")
    if _INSTALLED:
        return False
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    dispatch = CudaGraphManager.dispatch

    @wraps(dispatch)
    def exact_dispatch(self, num_reqs, num_tokens, *args, **kwargs):
        desc = dispatch(self, num_reqs, num_tokens, *args, **kwargs)
        if getattr(self, "_mach_2b_graph_mode", None) != "production":
            return desc
        return apply_descriptor(desc, num_tokens, num_reqs)

    CudaGraphManager.dispatch = exact_dispatch
    _INSTALLED = True
    return True


def inspect_worker(worker=None):
    if worker is None:
        return {"installed": _INSTALLED, "compilation_settings": compilation_settings()}
    return {"installed": _INSTALLED, **verify_capture(worker)}
