"""Frozen 2B BA adapter composed by the Mach champion profile."""
from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
from importlib import import_module
from importlib.metadata import version
import os
from pathlib import Path
from typing import Any

SHAPE = (32, 2048)
LAYERS = 18
KERNEL_SHA256 = "59756281b45716c0c5f25b18e331612de529e24a33cc4b37ec93efc7ab3e7379"
_INSTALLED = False
_GEMV: Any = None
_OP: Any = None
_COUNTS: dict[int, Counter] = {}
_PRECOMPILE_COUNT = 0


def enabled() -> bool:
    return os.environ.get("VLLM_MACH_PROFILE") == "qwen35-2b-mxfp8-champion-v1"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_kernel():
    """Resolve installed code only; never import the 4B profile."""
    if not enabled():
        raise RuntimeError("2B BA requires qwen35-2b-mxfp8-champion-v1")
    if version("b12x") != "1.2.6":
        raise RuntimeError("2B BA requires the audited b12x 1.2.6")
    kernel = import_module("vllm_mach.mxfp8.ba_gemv")
    if sha(kernel.__file__) != KERNEL_SHA256:
        raise RuntimeError("installed BA GEMV differs from the audited source")
    return kernel


def _small_impl(x: "torch.Tensor", weight: "torch.Tensor",
                private_weight: "torch.Tensor") -> "torch.Tensor":
    import torch
    m = int(x.shape[0])
    counts = _COUNTS.setdefault(m, Counter())
    capturing = torch.cuda.is_current_stream_capturing()
    phase = "capture" if capturing else "eager"
    reason = None
    if not 1 <= m <= 8:
        reason = "rows"
    elif (tuple(private_weight.shape) != SHAPE or private_weight.dtype != torch.bfloat16
          or not private_weight.is_contiguous() or private_weight.data_ptr() % 16):
        reason = "weight"
    else:
        if not x.is_contiguous() or x.data_ptr() % 16:
            counts[f"{phase}_input_contiguous_copies"] += 1
            x = x.contiguous()
        if x.data_ptr() % 16:
            reason = "input_alignment"
    launch = None if reason else _GEMV.get_cached_bf16_gemv_small_n(m, *SHAPE)
    if reason is None and launch is None:
        reason = "cache_miss"
    if reason:
        counts[f"{phase}_fallback_{reason}"] += 1
        return torch.nn.functional.linear(x, weight)
    y = torch.empty((m, SHAPE[0]), dtype=torch.bfloat16, device=x.device)
    launch(x, private_weight, y)
    # This count follows an actual cached kernel launch, never an attempted dispatch.
    counts[f"{phase}_kernel_calls"] += 1
    return y


def _fake(x: "torch.Tensor", weight: "torch.Tensor",
          private_weight: "torch.Tensor") -> "torch.Tensor":
    return x.new_empty((x.shape[0], weight.shape[0]))


def install(worker=None) -> bool:
    """After init_device, before load_model; only the exact 2B profile is accepted."""
    global _INSTALLED, _GEMV, _OP
    del worker
    if not enabled():
        raise RuntimeError("2B BA requires qwen35-2b-mxfp8-champion-v1")
    if _INSTALLED:
        return False
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if version("vllm").split("+", 1)[0] != "0.29.0":
        raise RuntimeError("2B BA adapter requires audited vLLM 0.29.0")
    for marker in ("_mx8_ba_only_installed", "_q35_ba_installed", "_two_b_ba_installed"):
        if getattr(UnquantizedLinearMethod, marker, False):
            raise RuntimeError(f"conflicting BA adapter: {marker}")
    _GEMV = load_kernel()
    # Resolve forward annotations for torch.library's schema inference.
    globals()["torch"] = torch
    _small_impl.__annotations__ = {name: torch.Tensor for name in
                                   ('x', 'weight', 'private_weight', 'return')}
    _OP = torch.library.custom_op("mach_2b_ba::small_n", mutates_args=())(_small_impl)
    _OP.register_fake(_fake)
    original_post = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply

    @wraps(original_post)
    def prepare(self, layer):
        global _PRECOMPILE_COUNT
        result = original_post(self, layer)
        weight = getattr(layer, "weight", None)
        if (weight is not None and tuple(weight.shape) == SHAPE
                and weight.dtype == torch.bfloat16 and weight.is_cuda
                and getattr(layer, "bias", None) is None):
            if getattr(layer, "_two_b_ba_private_weight", None) is not None:
                raise RuntimeError("2B BA weight prepared twice")
            private = weight.detach().clone().contiguous()
            if not torch.equal(private.view(torch.uint8), weight.contiguous().view(torch.uint8)):
                raise RuntimeError("2B BA private clone differs")
            _GEMV.precompile_bf16_gemv_small_n(private)
            if any(_GEMV.get_cached_bf16_gemv_small_n(m, *SHAPE) is None for m in range(1, 9)):
                raise RuntimeError("2B BA precompile missed a row count")
            layer._two_b_ba_private_weight = private
            _PRECOMPILE_COUNT += 1
        return result

    @wraps(original_apply)
    def apply(self, layer, x, bias=None):
        private = getattr(layer, "_two_b_ba_private_weight", None)
        if (private is None or x.ndim != 2 or x.shape[1] != SHAPE[1]
                or x.dtype != torch.bfloat16 or not x.is_cuda or bias is not None):
            return original_apply(self, layer, x, bias)
        # M stays symbolic in Dynamo; only the opaque op reads runtime rows.
        return _OP(x, layer.weight, private)

    UnquantizedLinearMethod.process_weights_after_loading = prepare
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._two_b_ba_installed = True
    _INSTALLED = True
    return True


def inspect_worker(worker) -> dict:
    import torch
    names, clone_bytes = [], 0
    for name, layer in worker.model_runner.model.named_modules():
        private = getattr(layer, "_two_b_ba_private_weight", None)
        if private is None:
            continue
        if not name.endswith("in_proj_ba") or tuple(private.shape) != SHAPE:
            raise RuntimeError(f"unexpected cloned projection: {name}")
        if (private.dtype != torch.bfloat16 or not private.is_contiguous()
                or private.data_ptr() % 16 or private.data_ptr() == layer.weight.data_ptr()
                or not torch.equal(private.view(torch.uint8), layer.weight.contiguous().view(torch.uint8))):
            raise RuntimeError(f"invalid/changed private BA weight: {name}")
        names.append(name)
        clone_bytes += private.numel() * private.element_size()
    if not _INSTALLED or len(names) != LAYERS or len(set(names)) != LAYERS:
        raise RuntimeError(f"2B BA requires {LAYERS} unique clones; got {len(names)}")
    return {"pid": os.getpid(), "enabled": enabled(), "installed": _INSTALLED,
            "ba_shape": list(SHAPE), "ba_layer_count": len(names), "ba_layers": sorted(names),
            "clone_bytes": clone_bytes, "clone_bitwise_equal": True,
            "precompile_invocations": _PRECOMPILE_COUNT,
            "counts_by_m": {str(m): dict(c) for m, c in sorted(_COUNTS.items())},
            "adapter_sha256": sha(__file__), "kernel_path": str(Path(_GEMV.__file__).resolve()),
            "kernel_sha256": sha(_GEMV.__file__), "b12x_version": version("b12x"),
            "vllm_version": version("vllm"),
            "receipt_semantics": "Actual eager/capture launches; graph replay does not increment Python counters",
            "arithmetic": "128-thread strided FP32 accumulation and reduction, then BF16; not cuBLAS bitwise equivalence"}


def verify_capture(worker, *, required_rows=(4,)) -> dict:
    receipt = inspect_worker(worker)
    if not required_rows or any(type(m) is not int or not 1 <= m <= 8 for m in required_rows):
        raise ValueError("required_rows must contain actual captured M in 1..8")
    missing = [m for m in required_rows if _COUNTS.get(m, {}).get("capture_kernel_calls", 0) == 0]
    bad = {m: dict(c) for m, c in _COUNTS.items() if 1 <= m <= 8
           and any("fallback" in key and value for key, value in c.items())}
    if missing or bad:
        raise RuntimeError(f"2B BA capture missing={missing}, small-M fallback={bad}")
    return receipt
