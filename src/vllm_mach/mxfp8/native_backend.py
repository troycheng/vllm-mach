# SPDX-License-Identifier: Apache-2.0

"""Optional native MXFP8 GEMM backend for vLLM 0.29.0 workers.

GEMM-only replacement of FlashInfer CUTLASS MXFP8. Activation quantization,
weight layout, bias, and reshape follow the vLLM class. Package integration
must call install() inside an EngineCore GPU worker after device setup and
before model compilation. Importing this module does not query CUDA or load
the native shared library.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch


_OP_NAME = "mach_mxfp8_native_gemm"
_DEFAULT_CONFIG = (-1, 1, 1, 0)
_POLICY = os.environ.get("MX8_NATIVE_POLICY", "all")
if _POLICY not in ("all", "wide-only"):
    raise ValueError(f"Invalid MX8_NATIVE_POLICY: {_POLICY}")
_LOCK = threading.RLock()
_INSTALLED = False
_REGISTERED = False
_NATIVE_LOADED = False
_NATIVE_IDENTITY: dict[str, Any] = {}
_SM120_DEVICES: frozenset[int] = frozenset()
_ORIGINAL_APPLY: Any = None
_TARGET_CLASS: Any = None
_COUNTS: Counter[str] = Counter()


@dataclass
class _WorkspaceEntry:
    workspace: torch.Tensor | None
    stream: Any  # Retain the stream wrapper so its handle is not recycled.


_WORKSPACES: dict[tuple[int, int, int, int, int, tuple[int, int, int, int]], _WorkspaceEntry] = {}


def _library_identity(library: Path, api: Any = None) -> dict[str, Any]:
    """Record the official artifact; profile release locks are checked elsewhere.

    The official wheel does not promise an experiment COMPLETE.json. Missing
    source metadata stays explicitly absent rather than inventing provenance.
    """
    try:
        runtime_version = version("mxfp6-sm120")
    except PackageNotFoundError:
        runtime_version = None
    candidates = [library.parent / "metadata.json"]
    if api is not None and getattr(api, "__file__", None):
        candidates.insert(0, Path(api.__file__).parent / "metadata.json")
    metadata_path = next((path for path in candidates if path.is_file()), None)
    metadata = None if metadata_path is None else json.loads(metadata_path.read_text())
    return {"path": str(library),
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "mxfp6_version": runtime_version,
            "metadata_path": None if metadata_path is None else str(metadata_path),
            "metadata": metadata}


def _verify_schema(*, require_dual: bool = False) -> None:
    expected = {
        "gemm": ("Tensor a", "Tensor b", "Tensor sa", "Tensor sb",
                 "Tensor(a!)? out", "Tensor? workspace", "int tactic",
                 "int splits", "int swizzle", "int sms"),
        "allocate_workspace": ("Tensor a", "Tensor b", "Tensor sa",
                               "Tensor sb", "Tensor out", "int tactic",
                               "int splits", "int swizzle", "int sms"),
    }
    if require_dual:
        expected["dual_gemm_out"] = (
            "Tensor hi", "Tensor weight", "Tensor s_hi", "Tensor s_weight",
            "Tensor res", "Tensor s_res", "Tensor(a!) out",
        )
    for name, fragments in expected.items():
        try:
            schema = str(getattr(torch.ops.mxfp8_sm120, name).default._schema)
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(f"MXFP8 native operator missing: {name}") from exc
        if not all(fragment in schema for fragment in fragments):
            raise RuntimeError(f"MXFP8 native schema mismatch for {name}: {schema}")


def _load_native() -> None:
    global _NATIVE_LOADED, _NATIVE_IDENTITY
    if _NATIVE_LOADED:
        return
    # Explicit binary path supports a standalone deployment; the unified
    # mxfp6-sm120 wheel provides its own library discovery as the fallback.
    api = None
    override = os.environ.get("MXFP8_LIBRARY_PATH")
    if override:
        library = Path(override).expanduser().resolve()
        if not library.is_file():
            raise ImportError(f"MXFP8_LIBRARY_PATH does not exist: {library}")
        torch.ops.load_library(str(library))
    else:
        try:
            api = importlib.import_module("mxfp6.mxfp8")
        except ImportError as exc:
            raise ImportError(
                "MXFP8 native library not found: install the release unified "
                "wheel or set MXFP8_LIBRARY_PATH to its built native library"
            ) from exc
        # The official loader returns the actual resolved artifact path.
        library = Path(api.load_library()).resolve()
    _verify_schema()
    _NATIVE_IDENTITY = _library_identity(library, api)
    _NATIVE_LOADED = True


def verify_runtime(*, require_dual: bool = False) -> None:
    """Check the build identity and schemas after worker device setup."""
    with _LOCK:
        _load_native()
        if require_dual:
            _verify_schema(require_dual=True)


def _native_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # This opaque custom-op body is the only runtime call path that records
    # counts or touches the workspace lock. Dynamo sees only _native_fake.
    m, k = activation.shape  # Read runtime M inside the opaque op.
    n = weight.shape[0]
    device_index = activation.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device_index)
    key = (device_index, stream.cuda_stream, m, n, k, _DEFAULT_CONFIG)
    capturing = torch.cuda.is_current_stream_capturing()

    with _LOCK:
        entry = _WORKSPACES.get(key)
        if entry is None and key not in _WORKSPACES:
            if capturing:
                _COUNTS["capture_cache_miss"] += 1
                raise RuntimeError(
                    "MXFP8 native workspace was not warmed on this CUDA "
                    "stream and shape before graph capture"
                )
            _load_native()
            # This same fresh output is used for the current GEMM. Querying
            # with an old/layer-global output would hide output aliasing.
            output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
            workspace = torch.ops.mxfp8_sm120.allocate_workspace(
                activation,
                weight,
                activation_scale,
                weight_scale,
                output,
                *_DEFAULT_CONFIG,
            )
            entry = _WorkspaceEntry(workspace=workspace, stream=stream)
            _WORKSPACES[key] = entry
            _COUNTS["workspace_miss"] += 1
        else:
            output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
            _COUNTS["workspace_hit"] += 1

    # CUDA stream order serializes reuse within a key. Separate stream keys
    # own separate workspaces. The native kernel resets Stream-K barriers.
    torch.ops.mxfp8_sm120.gemm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        output,
        entry.workspace,
        *_DEFAULT_CONFIG,
    )
    with _LOCK:
        _COUNTS["native_calls"] += 1
        _COUNTS["native_capture_calls" if capturing else "native_eager_calls"] += 1
    return output


def _native_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del activation_scale, weight_scale
    # Keep M symbolic. Never convert activation.shape[0] to a Python int here.
    return torch.empty(
        (activation.shape[0], weight.shape[0]),
        device=activation.device,
        dtype=torch.bfloat16,
    )


def _register_op() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_native_impl,
        mutates_args=[],
        fake_impl=_native_fake,
    )
    _REGISTERED = True


def gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Call the opaque native op with FlashInfer-prepared MXFP8 operands.

    Callers must install this backend in the GPU worker and preserve the
    selected kernel's static dtype/layout/device guards. This entry point
    keeps runtime M and workspace ownership inside the custom op, including
    for profile overlays that delegate their unchanged GEMM path here.
    """
    return getattr(torch.ops.vllm, _OP_NAME)(
        activation, weight, activation_scale, weight_scale
    )


def _fallback(self: Any, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return _ORIGINAL_APPLY(self, layer, x, bias)


def _apply_weights(self: Any, layer: Any, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    # Only static shape/dtype/device properties are inspected before the
    # opaque op. In particular, M is not converted to a Python constant.
    weight = layer.weight
    weight_scale = layer.weight_scale
    if x.dtype != torch.bfloat16:
        return _fallback(self, layer, x, bias)
    if x.device.type != "cuda" or weight.device != x.device or weight_scale.device != x.device:
        return _fallback(self, layer, x, bias)
    if x.device.index not in _SM120_DEVICES:
        return _fallback(self, layer, x, bias)
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        return _fallback(self, layer, x, bias)
    n, k = weight.shape
    if _POLICY == "wide-only" and n == 2560:
        return _fallback(self, layer, x, bias)
    if n < 128 or k < 128 or n % 128 or k % 128:
        return _fallback(self, layer, x, bias)
    if weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous():
        return _fallback(self, layer, x, bias)
    if not weight_scale.is_contiguous() or weight_scale.element_size() != 1:
        return _fallback(self, layer, x, bias)
    if weight_scale.numel() < n * (k // 32):
        return _fallback(self, layer, x, bias)

    # These are the FlashInfer CUTLASS apply steps from vLLM 0.29.0. Its
    # process_weights_after_loading remains untouched, including scale swizzle.
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    input_shape = x.shape
    input_2d = x.view(-1, k)
    activation, activation_scale = mxfp8_e4m3_quantize(
        input_2d, is_sf_swizzled_layout=True
    )
    output = gemm(activation, weight, activation_scale, weight_scale)
    if bias is not None:
        output = output + bias
    return output.view((*input_shape[:-1], n))


def install() -> bool:
    """Patch the already selected FlashInfer CUTLASS kernel class in this worker."""
    global _INSTALLED, _ORIGINAL_APPLY, _TARGET_CLASS, _SM120_DEVICES
    with _LOCK:
        if _INSTALLED:
            return False
        module = importlib.import_module("vllm.model_executor.kernels.linear.mxfp8.flashinfer")
        cls = module.FlashInferCutlassMxfp8LinearKernel
        _register_op()
        # Device capability is stable for this worker and must not be queried
        # from the method Dynamo traces for every linear call.
        _SM120_DEVICES = frozenset(
            index
            for index in range(torch.cuda.device_count())
            if torch.cuda.get_device_capability(index) == (12, 0)
        )
        _ORIGINAL_APPLY = cls.apply_weights
        _TARGET_CLASS = cls
        cls.apply_weights = _apply_weights
        _INSTALLED = True
        _COUNTS["installs"] += 1
        return True


def uninstall() -> bool:
    """Restore the class method in this worker; op registration remains."""
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            return False
        _TARGET_CLASS.apply_weights = _ORIGINAL_APPLY
        _INSTALLED = False
        _COUNTS["uninstalls"] += 1
        return True


def inspect(worker: Any = None) -> dict[str, Any]:
    """Return this process's counters; invoke inside each vLLM worker.

    collective_rpc may inject the worker object; only its scalar identity is
    returned. This function performs no RPC.
    """
    rank = getattr(worker, "rank", None)
    local_rank = getattr(worker, "local_rank", None)
    worker_info = {
        "type": None if worker is None else type(worker).__name__,
        "rank": rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
        "local_rank": local_rank if isinstance(local_rank, int) and not isinstance(local_rank, bool) else None,
    }
    with _LOCK:
        entries = [
            {
                "device": key[0],
                "stream": key[1],
                "m": key[2],
                "n": key[3],
                "k": key[4],
                "config": key[5],
                "workspace_bytes": (0 if value.workspace is None else
                                    value.workspace.numel() * value.workspace.element_size()),
            }
            for key, value in _WORKSPACES.items()
        ]
        return {
            "worker": worker_info,
            "pid": os.getpid(),
            "installed": _INSTALLED,
            "policy": _POLICY,
            "op_registered": _REGISTERED,
            "binary": dict(_NATIVE_IDENTITY),
            "sm120_devices": sorted(_SM120_DEVICES),
            "counts": dict(_COUNTS),
            "workspaces": entries,
            "workspace_bytes_total": sum(e["workspace_bytes"] for e in entries),
        }


__all__ = ["gemm", "inspect", "install", "uninstall", "verify_runtime"]
