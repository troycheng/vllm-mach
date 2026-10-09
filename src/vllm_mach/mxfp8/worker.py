# SPDX-License-Identifier: Apache-2.0

"""Install the opt-in MXFP8 GEMM backend inside vLLM's GPU worker.

The plugin may execute in an API/EngineCore parent. Only the wrapper is
installed there; native library loading and CUDA capability queries occur
after Worker.init_device() runs in the actual EngineCore GPU worker.
"""

from __future__ import annotations

from functools import wraps
from importlib import import_module


def install_worker_backend(worker=None) -> bool:
    """Install the native backend after this GPU worker's device setup.

    Full model profiles can call this directly from their own ordered
    lifecycle hook before loading/compiling the model. No parent-process
    plugin registration should call this function.
    """
    del worker
    backend = import_module("vllm_mach.mxfp8.native_backend")
    backend.verify_runtime()
    return backend.install()


def verify_worker_execution(worker=None, *, require_capture: bool = False) -> dict:
    """Validate real native execution and return this worker's receipt.

    Graph-disabled native deployments may use eager calls alone. Complete
    profiles with CUDA Graphs must pass ``require_capture=True`` after warmup.
    Capture calls demonstrate recording; they do not count graph replays.
    """
    backend = import_module("vllm_mach.mxfp8.native_backend")
    receipt = backend.inspect(worker)
    counts = receipt["counts"]
    if not receipt["installed"] or counts.get("native_calls", 0) == 0:
        raise RuntimeError(
            "MXFP8 native opt-in selected but no eligible native GEMM ran; "
            "check FlashInfer CUTLASS kernel selection and dtype/shape guards"
        )
    if counts.get("capture_cache_miss", 0):
        raise RuntimeError("MXFP8 native CUDA graph capture missed warmed workspace")
    if require_capture and counts.get("native_capture_calls", 0) == 0:
        raise RuntimeError("MXFP8 native CUDA graph capture recorded no native GEMM")
    return receipt


def install_worker_hook() -> bool:
    """Register the standalone native opt-in hooks without touching CUDA."""
    from vllm.v1.worker.gpu_worker import Worker

    if (getattr(Worker, "_mach_2b_hook", False)
            or getattr(Worker, "_mach_mxfp8_champion_hook", False)):
        raise RuntimeError("A complete MXFP8 profile is already installed")
    if getattr(Worker, "_mach_mxfp8_native_hook", False):
        return False

    original_init = Worker.init_device
    original_compile = Worker.compile_or_warm_up_model

    @wraps(original_init)
    def init_device(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        install_worker_backend(self)
        return result

    @wraps(original_compile)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original_compile(self, *args, **kwargs)
        verify_worker_execution(self)
        return result

    Worker.init_device = init_device
    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    Worker._mach_mxfp8_native_hook = True
    return True


__all__ = ["install_worker_backend", "install_worker_hook", "verify_worker_execution"]
