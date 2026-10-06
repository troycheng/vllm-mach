# SPDX-License-Identifier: Apache-2.0
"""Exact2048 PIECEWISE policy; no parent CUDA work or diagnostic GPU counters."""
from collections import Counter
from functools import wraps

BASE = (1, 2, 4, *range(8, 257, 8))
CAPTURE_SIZES = (*BASE, 2048)
FULL_SIZES = tuple(x for x in BASE if x <= 128)
MAX_CAPTURE_SIZE = 2048
_READY = False
_MANAGER = None
_ROUTES = Counter()


def compilation_settings():
    """Fields to merge into the complete profile's compilation configuration."""
    return {"cudagraph_capture_sizes": list(CAPTURE_SIZES),
            "max_cudagraph_capture_size": MAX_CAPTURE_SIZE}


def apply_descriptor(desc, num_tokens, num_reqs):
    """Correct large tail descriptors before the runner pads its input tensors."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    if num_tokens is None or num_reqs is None:
        raise RuntimeError("PW2048 dispatch requires actual token and request counts")
    if num_tokens > 256 and num_tokens != 2048 and desc.cg_mode == CUDAGraphMode.PIECEWISE:
        desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,num_reqs=num_reqs,num_active_loras=desc.num_active_loras)
    if num_tokens > 256 and desc.num_tokens != num_tokens:
        raise RuntimeError("Non-target padding changed arithmetic shape")
    if num_tokens == 2048 and desc.cg_mode != CUDAGraphMode.PIECEWISE:
        raise RuntimeError("actual2048 requires PIECEWISE2048")
    return desc


def install_dispatch():
    """Register before compilation; enable routing after verified graph capture."""
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    if getattr(CudaGraphManager, "_mach_mxfp8_pw2048", False):
        return False
    original = CudaGraphManager.dispatch
    @wraps(original)
    def dispatch(self, *args, **kwargs):
        desc = original(self, *args, **kwargs)
        if not _READY:
            return desc
        num_tokens = kwargs.get("num_tokens", args[1] if len(args) > 1 else None)
        num_reqs = kwargs.get("num_reqs", args[0] if args else None)
        desc = apply_descriptor(desc, num_tokens, num_reqs)
        _ROUTES[(desc.cg_mode.name, int(num_tokens), int(desc.num_tokens))] += 1
        return desc
    CudaGraphManager.dispatch = dispatch
    CudaGraphManager._mach_mxfp8_pw2048 = True
    return True


def verify_capture(worker):
    """After compile: bind the manager and prove the exact capture/capacity policy."""
    global _READY, _MANAGER
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
    native = native_backend.inspect(worker)
    if native["counts"].get("capture_cache_miss", 0):
        raise RuntimeError("native MXFP8 workspace missed during graph capture")
    _MANAGER, _READY = manager, True
    return inspect_worker(worker)


def inspect_worker(worker=None):
    return {"ready": _READY, "capture_sizes": list(CAPTURE_SIZES),
            "full_sizes": list(FULL_SIZES), "max_capture_size": MAX_CAPTURE_SIZE,
            "route_counts": [{"mode": k[0], "actual_tokens": k[1],
                              "physical_tokens": k[2], "calls": v}
                             for k,v in sorted(_ROUTES.items())],
            "descriptor_counts": {mode.name: len(descs) for mode,descs in
                                  getattr(_MANAGER, "_capture_descs", {}).items()},
            "counts_are_python_construction_only": True}


def install_hook():
    """Optional composed wrapper; complete profiles may use the lower functions."""
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, "_mach_mxfp8_pw2048_hook", False):
        return False
    install_dispatch()
    original = Worker.compile_or_warm_up_model
    @wraps(original)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        verify_capture(self)
        return result
    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    Worker._mach_mxfp8_pw2048_hook = True
    return True
