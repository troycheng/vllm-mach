# SPDX-License-Identifier: Apache-2.0
"""Install explicit FP32 FP8-KV scales after load_model, before profiling.

Profile metadata contains only the selected model's scale values and module
names. Asset provenance/identity is checked by the profile loader; this module
never requires a calibration dataset, experiment receipt or private path.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import struct

PROFILE = "qwen35-4b-mxfp8-champion-v1"
LAYERS = tuple(range(3, 32, 4))
_STATE = "_mach_mxfp8_kv_scale_setup"


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def load_profile(profile, *, model_dir=None):
    """Read a metadata mapping, JSON path or model directory/mach_profile.json.

    The required fields are profile and kv_scales. Each of the eight rows has
    layer_index, module_name, k_scale and v_scale; both scales are already FP32.
    Extra metadata is left to the model asset validator and not copied here.
    """
    if profile is None:
        if model_dir is None:
            raise ValueError("Provide FP8 KV profile metadata or a model directory")
        profile = Path(model_dir) / "mach_profile.json"
    if not isinstance(profile, Mapping):
        path = Path(profile)
        if path.is_dir():
            path = path / "mach_profile.json"
        profile = json.loads(path.read_text())
    if not isinstance(profile, Mapping) or profile.get("profile") != PROFILE:
        raise ValueError(f"FP8 KV metadata requires profile={PROFILE}")
    rows = profile.get("kv_scales")
    if not isinstance(rows, list) or len(rows) != len(LAYERS):
        raise ValueError("FP8 KV profile requires eight scale rows")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FP8 KV scale rows must be objects")
        index = row.get("layer_index")
        if (type(index) is not int or index not in LAYERS or index in seen
                or row.get("module_name") != f"model.layers.{index}.self_attn.attn"):
            raise ValueError("FP8 KV layer indices/module names differ or are duplicated")
        seen.add(index)
        checked = {"layer_index": index, "module_name": row["module_name"]}
        for limb in ("k", "v"):
            value = row.get(limb + "_scale")
            if type(value) not in (int, float):
                raise ValueError("FP8 KV scales must be finite positive FP32 numbers")
            try:
                value = float(value)
                valid = math.isfinite(value) and value > 0 and value == _f32(value)
            except (OverflowError, struct.error):
                valid = False
            if not valid:
                raise ValueError("FP8 KV scales must be finite positive, exactly FP32")
            checked[limb + "_scale"] = value
        result.append(checked)
    return {"profile": PROFILE, "kv_scales": sorted(result, key=lambda row: row["layer_index"])}


def _modules(worker):
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    return {name: module for name, module in worker.model_runner.model.named_modules()
            if isinstance(getattr(module, "impl", None), FlashInferImpl)}


def _validate_module(module):
    impl = module.impl
    if module.kv_cache_dtype != "fp8_e4m3" or impl.cache_dtype != "fp8_e4m3":
        raise RuntimeError("Actual FlashInfer FP8 KV dtype differs")
    if (impl.num_heads, impl.num_kv_heads, impl.head_size) != (16, 4, 256):
        raise RuntimeError("Actual full-attention geometry differs")


def _validate_mirrors(module):
    import torch
    for limb in ("k", "v"):
        for suffix in ("_scale", "_scale_cpu"):
            value = getattr(module, "_" + limb + suffix, None)
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise RuntimeError(f"Expected singleton scale mirror: {limb}{suffix}")


def install_scale_mirrors(module, limb, value, device):
    """Create device FP32, host FP32 and Python float owners before capture."""
    import torch
    previous = {}
    for suffix in ("_scale", "_scale_cpu"):
        old = getattr(module, "_" + limb + suffix)
        if not isinstance(old, torch.Tensor) or old.numel() != 1:
            raise RuntimeError("Expected singleton scale mirror: " + limb + suffix)
        previous[suffix] = {"device": str(old.device), "dtype": str(old.dtype), "shape": list(old.shape)}
    old = getattr(module, "_" + limb + "_scale")
    if old.device == torch.device(device) and old.dtype == torch.float32:
        old.fill_(value)
    else:
        setattr(module, "_" + limb + "_scale", torch.tensor(value, dtype=torch.float32, device=device))
    # Explicit CPU placement also fixes mirrors created under a CUDA default
    # device context. The installer has established that no FI cache exists.
    setattr(module, "_" + limb + "_scale_cpu", torch.tensor(value, dtype=torch.float32, device="cpu"))
    setattr(module, "_" + limb + "_scale_float", value)
    return previous


def install_scales(worker, profile=None):
    """Install once after Worker.load_model, before profile/warmup/capture."""
    import torch
    if hasattr(worker, _STATE):
        raise RuntimeError("FP8 KV scales are already installed")
    if worker.vllm_config.cache_config.cache_dtype != "fp8_e4m3":
        raise RuntimeError("Actual worker FP8 KV configuration is absent")
    if torch.device(worker.device).type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Install FP8 KV scales on the CUDA worker before capture")
    metadata = load_profile(profile, model_dir=worker.vllm_config.model_config.model)
    modules = _modules(worker)
    if set(modules) != {row["module_name"] for row in metadata["kv_scales"]}:
        raise RuntimeError("Actual eight FlashInfer module names differ from FP8 KV profile")
    # Validate every layer before mutating any scale, so a late coverage/cache
    # failure cannot leave a partially configured worker.
    for module in modules.values():
        _validate_module(module)
        if module.impl.bmm1_scale is not None or module.impl.bmm2_scale is not None:
            raise RuntimeError("FlashInfer scale cache initialized before sidecar installation")
        _validate_mirrors(module)
    records = []
    with torch.no_grad():
        for row in metadata["kv_scales"]:
            module = modules[row["module_name"]]
            previous = {limb: install_scale_mirrors(module, limb, row[limb + "_scale"], worker.device)
                        for limb in ("k", "v")}
            records.append({"name": row["module_name"], "previous_mirrors": previous,
                            "bmm1_before": None, "bmm2_before": None,
                            "installed_before_profile_capture": True})
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    setattr(worker, _STATE, {"metadata": metadata, "metadata_sha256": digest, "rows": records})
    return inspect_scales(worker, require_storage=False)


def inspect_scales(worker, *, require_storage=True):
    """Check frozen mirrors and, after cache allocation, actual FP8 storage."""
    import torch
    setup = getattr(worker, _STATE, None)
    if setup is None:
        return {"installed": False}
    if worker.vllm_config.cache_config.cache_dtype != "fp8_e4m3":
        raise RuntimeError("Worker FP8 KV configuration changed")
    modules = _modules(worker)
    expected = {row["module_name"]: row for row in setup["metadata"]["kv_scales"]}
    if set(modules) != set(expected):
        raise RuntimeError("FlashInfer module coverage changed after scale installation")
    layers = []
    for name, row in expected.items():
        module = modules[name]
        _validate_module(module)
        impl = module.impl
        mirrors = {}
        for limb in ("k", "v"):
            device_scale = getattr(module, "_" + limb + "_scale")
            host_scale = getattr(module, "_" + limb + "_scale_cpu")
            float_scale = getattr(module, "_" + limb + "_scale_float")
            if (device_scale.dtype != torch.float32 or device_scale.device != torch.device(worker.device)
                    or device_scale.numel() != 1 or host_scale.dtype != torch.float32
                    or host_scale.device.type != "cpu" or host_scale.numel() != 1):
                raise RuntimeError("FP32 device/CPU scale owners changed")
            values = [float(device_scale.item()), float(float_scale), float(host_scale.item())]
            if values != [row[limb + "_scale"]] * 3:
                raise RuntimeError("Frozen FP8 KV scale mirrors changed")
            mirrors[limb] = values
        if impl.bmm2_scale is not None and impl.bmm2_scale != row["v_scale"]:
            raise RuntimeError("FlashInfer cached V scale changed")
        cache = module.kv_cache
        bound = isinstance(cache, torch.Tensor) and cache.numel() > 0
        if require_storage and (not bound or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
                                or cache.device != torch.device(worker.device)):
            raise RuntimeError("Actual FP8 KV storage is not bound to this worker")
        layers.append({"name": name, "layer_index": row["layer_index"],
                       "logical_dtype": impl.cache_dtype,
                       "geometry": [impl.num_heads, impl.num_kv_heads, impl.head_size],
                       "scale_mirrors_device_float_cpu": mirrors,
                       "bmm1_scale": impl.bmm1_scale, "bmm2_scale": impl.bmm2_scale,
                       "kv_dtype": str(cache.dtype) if bound else None,
                       "kv_shape": list(cache.shape) if bound else None,
                       "kv_stride": list(cache.stride()) if bound else None,
                       "kv_ptr": cache.data_ptr() if bound else None})
    return {"installed": True, "profile": PROFILE, "metadata_sha256": setup["metadata_sha256"],
            "require_storage": require_storage, "layers": layers, "installation": setup["rows"]}


__all__ = ["load_profile", "install_scales", "inspect_scales"]
