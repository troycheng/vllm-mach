# SPDX-License-Identifier: Apache-2.0

"""Champion BA-only BF16 small-N GEMV with the original cuBLAS fallback.

Install after device initialization and before model loading. The exact
64x2560 BF16/no-bias projection gets a fresh private weight clone. Runtime
M1..8 dispatch stays inside the historical mx8_ba_only::small_n opaque op.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps
from importlib import import_module
from importlib.metadata import version
import os
from typing import Any

import torch

_INSTALLED = False
_GEMV: Any = None
_PRECOMPILE_COUNT = 0
_COUNTS: dict[int, Counter] = {}
_TAGGED_NAMES: tuple[str, ...] = ()
_SMALL_OP: Any = None


def _small_impl(x: torch.Tensor, weight: torch.Tensor,
                private_weight: torch.Tensor) -> torch.Tensor:
    m = int(x.shape[0])
    bucket = _COUNTS.setdefault(m, Counter())
    if 1 <= m <= 8:
        bucket["normal_capture_calls" if torch.cuda.is_current_stream_capturing()
               else "normal_eager_calls"] += 1
        return _GEMV.bf16_gemv_small_n(x, private_weight)
    bucket["fallback_calls"] += 1
    return torch.nn.functional.linear(x, weight)


def _fake(x: torch.Tensor, weight: torch.Tensor,
          private_weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def install(worker=None) -> bool:
    """Install shape-restricted weight preparation and apply in this GPU worker."""
    global _INSTALLED, _GEMV, _SMALL_OP
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    del worker
    if _INSTALLED:
        return False
    if getattr(UnquantizedLinearMethod, "_mx8_ba_only_installed", False) or getattr(
            UnquantizedLinearMethod, "_q35_ba_installed", False):
        raise RuntimeError("a conflicting BA adapter is already installed")
    installed = version("b12x")
    if installed != "1.2.6":
        raise RuntimeError(f"champion BA requires b12x 1.2.6; found {installed}")
    _GEMV = import_module("vllm_mach.mxfp8.ba_gemv")
    _SMALL_OP = torch.library.custom_op("mx8_ba_only::small_n", mutates_args=())(_small_impl)
    _SMALL_OP.register_fake(_fake)
    original_post = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply

    @wraps(original_post)
    def prepare(self, layer):
        global _PRECOMPILE_COUNT
        result = original_post(self, layer)
        weight = getattr(layer, "weight", None)
        if (weight is not None and tuple(weight.shape) == (64, 2560)
                and weight.dtype == torch.bfloat16 and weight.is_cuda
                and getattr(layer, "bias", None) is None):
            if getattr(layer, "_mx8_ba_private_weight", None) is not None:
                raise RuntimeError("BA weight prepared twice")
            private = weight.detach().clone().contiguous()
            if not torch.equal(private, weight):
                raise AssertionError("BA private weight clone differs")
            layer._mx8_ba_private_weight = private
            _GEMV.precompile_bf16_gemv_small_n(private)
            _PRECOMPILE_COUNT += 1
        return result

    @wraps(original_apply)
    def apply(self, layer, x, bias=None):
        private = getattr(layer, "_mx8_ba_private_weight", None)
        if (private is None or x.ndim != 2 or x.shape[1] != 2560
                or x.dtype != torch.bfloat16 or not x.is_cuda or bias is not None):
            return original_apply(self, layer, x, bias)
        return _SMALL_OP(x, layer.weight, private)

    UnquantizedLinearMethod.process_weights_after_loading = prepare
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._mx8_ba_only_installed = True
    _INSTALLED = True
    return True


def inspect_worker(worker) -> dict[str, Any]:
    """Check all loaded BA clones and report eager/capture counts, without RPC."""
    global _TAGGED_NAMES
    rows, clone_bytes = [], 0
    for name, layer in worker.model_runner.model.named_modules():
        private = getattr(layer, "_mx8_ba_private_weight", None)
        if private is None:
            continue
        if not name.endswith("in_proj_ba"):
            raise RuntimeError(f"non-BA projection was cloned: {name}")
        if tuple(private.shape) != (64, 2560) or private.dtype != torch.bfloat16:
            raise RuntimeError(f"invalid BA private weight: {name}")
        if not torch.equal(private, layer.weight):
            raise RuntimeError(f"BA private weight changed after loading: {name}")
        rows.append(name)
        clone_bytes += private.numel() * private.element_size()
    if len(rows) != 24 or len(set(rows)) != 24:
        raise RuntimeError(f"champion BA requires 24 unique layers; got {len(rows)}")
    _TAGGED_NAMES = tuple(sorted(rows))
    return {"mode": "normal", "pid": os.getpid(), "installed": _INSTALLED,
            "ba_layers": list(_TAGGED_NAMES), "ba_layer_count": len(rows),
            "clone_bytes": clone_bytes, "weight_clone_bitwise": True,
            "precompile_invocations": _PRECOMPILE_COUNT, "small_m_max": 8,
            "b12x_version": version("b12x"),
            "capture_hits_at_write": sum(counts["normal_capture_calls"]
                                         for counts in _COUNTS.values()),
            "counts_by_m": {str(m): dict(counts) for m, counts in sorted(_COUNTS.items())},
            "receipt_semantics": "Python eager/capture calls, excludes graph replay"}


def prepare_model(worker) -> dict[str, Any]:
    """Check post-load clones; precompilation already happened during loading."""
    if not _INSTALLED:
        raise RuntimeError("install BA before loading the model")
    return inspect_worker(worker)


def verify_capture(worker, *, required_rows=(1, 2, 4, 8)) -> dict[str, Any]:
    receipt = inspect_worker(worker)
    missing = [m for m in required_rows
               if _COUNTS.get(m, {}).get("normal_capture_calls", 0) == 0]
    if not _INSTALLED or missing:
        raise RuntimeError(f"champion BA graph capture missed rows: {missing}")
    return receipt


__all__ = ["install", "prepare_model", "inspect_worker", "verify_capture"]
