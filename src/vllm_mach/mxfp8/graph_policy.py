# SPDX-License-Identifier: Apache-2.0
"""Capture settings and replay verification for the 4B MXFP8 profile."""
from functools import wraps

BASE = (1, 2, 4, *range(8, 257, 8))
CAPTURE_SIZES = (*BASE, 2048)
FULL_SIZES = tuple(x for x in BASE if x <= 128)
MAX_CAPTURE_SIZE = 2048
_READY = False
_MANAGER = None


def compilation_settings():
    """Fields to merge into the complete profile's compilation configuration."""
    return {"cudagraph_capture_sizes": list(CAPTURE_SIZES),
            "max_cudagraph_capture_size": MAX_CAPTURE_SIZE}


def verify_capture(worker):
    """After compile: bind the manager and prove the exact capture/capacity policy."""
    global _READY, _MANAGER
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config.compilation import CUDAGraphMode
    from . import native_backend
    manager = worker.model_runner.cudagraph_manager
    if manager is None or not manager._graphs_captured:
        raise RuntimeError("PW2048 manager did not finish graph capture")
    if _READY and manager is not _MANAGER:
        raise RuntimeError("PW2048 manager changed after capture")
    actual = {mode.name: sorted(d.num_tokens for d in descs)
              for mode, descs in manager._capture_descs.items()}
    if actual.get("PIECEWISE") != list(CAPTURE_SIZES) or actual.get("FULL") != list(FULL_SIZES):
        raise RuntimeError(f"PW2048 capture descriptors differ: {actual}")
    cfg = worker.vllm_config.compilation_config
    if cfg.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE or cfg.max_cudagraph_capture_size != 2048:
        raise RuntimeError("PW2048 requires FULL_AND_PIECEWISE and maxcapture2048")
    if worker.vllm_config.cache_config.kv_cache_memory_bytes != 19*2**30:
        raise RuntimeError("PW2048 altered fixed 19 GiB KV capacity")
    prefill_graphs = [
        entry.cudagraph
        for wrapper in list(CUDAGraphWrapper._all_instances)
        if wrapper.vllm_config is worker.vllm_config
        and wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
        for desc, entry in wrapper.concrete_cudagraph_entries.items()
        if desc.num_tokens == MAX_CAPTURE_SIZE
    ]
    if not prefill_graphs or any(graph is None for graph in prefill_graphs):
        raise RuntimeError("Missing 4B PIECEWISE prefill capture")
    for tokens in (257, 1728, 1848, 1920, 2040, MAX_CAPTURE_SIZE):
        desc = manager.dispatch(num_reqs=1, num_tokens=tokens,
                                uniform_token_count=None, num_active_loras=0)
        if desc.cg_mode != CUDAGraphMode.PIECEWISE or desc.num_tokens != MAX_CAPTURE_SIZE:
            raise RuntimeError(f"Missing padded prefill replay at {tokens} tokens")
    native = native_backend.inspect(worker)
    if native["counts"].get("capture_cache_miss", 0):
        raise RuntimeError("native MXFP8 workspace missed during graph capture")
    _MANAGER, _READY = manager, True
    return inspect_worker(worker)


def inspect_worker(worker=None):
    return {"ready": _READY, "capture_sizes": list(CAPTURE_SIZES),
            "full_sizes": list(FULL_SIZES), "max_capture_size": MAX_CAPTURE_SIZE,
            "dispatch": "vllm", "route_counts": [],  # No per-step Mach hook.
            "descriptor_counts": {mode.name: len(descs) for mode,descs in
                                  getattr(_MANAGER, "_capture_descs", {}).items()},
            "counts_are_python_construction_only": True}


def install_hook():
    """Optional composed wrapper; complete profiles may use the lower functions."""
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, "_mach_mxfp8_pw2048_hook", False):
        return False
    original = Worker.compile_or_warm_up_model
    @wraps(original)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        verify_capture(self)
        return result
    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    Worker._mach_mxfp8_pw2048_hook = True
    return True
