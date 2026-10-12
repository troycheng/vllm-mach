# SPDX-License-Identifier: Apache-2.0
"""Validate native vLLM BF16 SSM pools without replacing GDN or scheduler ops."""
import os

_STATE = "_mach_bf16_stock_gdn"


def _layers(worker):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    return [m for m in worker.get_model().modules()
            if isinstance(m, QwenGatedDeltaNetAttention)]


def _check_unpatched():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as GDN
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    for cls, flags in ((GDN, ("_mach_mxfp8_ordered_installed", "_two_b_gdn_ordered_installed")),
                       (GPUModelRunner, ("_mach_mxfp8_ordered_lifecycle", "_two_b_gdn_ordered_lifecycle"))):
        if any(getattr(cls, flag, False) for flag in flags):
            raise RuntimeError("BF16 stock GDN cannot coexist with ordered GDN hooks")


def install_runtime(worker):
    _check_unpatched()
    if worker.vllm_config.cache_config.mamba_ssm_cache_dtype != "bfloat16":
        raise RuntimeError("BF16 stock GDN requires explicit bfloat16 SSM configuration")
    if hasattr(worker, _STATE):
        raise RuntimeError("Stock GDN lifecycle already initialized")
    setattr(worker, _STATE, dict(ready=False, cache_initialized=False, layers=[]))


def prepare_layers(worker):
    state = getattr(worker, _STATE)
    layers = _layers(worker)
    expected = 18 if worker.vllm_config.model_config.hf_text_config.hidden_size == 2048 else 24
    if state["layers"] or len(layers) != expected or len({m.prefix for m in layers}) != expected:
        raise RuntimeError("Unexpected stock GDN layer set")
    import torch
    if any(m.get_state_dtype()[1] != torch.bfloat16 for m in layers):
        raise RuntimeError("Model did not select BF16 recurrent state")
    state["layers"] = [m.prefix for m in layers]


def _inspect_pools(worker):
    import torch
    _check_unpatched()
    state = getattr(worker, _STATE)
    layers = _layers(worker)
    if not state["layers"] or [m.prefix for m in layers] != state["layers"]:
        raise RuntimeError("Stock GDN layer identities changed")
    pools = []
    for layer in layers:
        if len(layer.kv_cache) != 2:
            raise RuntimeError(f"Unbound stock GDN cache: {layer.prefix}")
        conv, recurrent = layer.kv_cache
        if (conv.dtype != torch.bfloat16 or recurrent.dtype != torch.bfloat16
                or not conv.is_cuda or not recurrent.is_cuda or recurrent.ndim != 4):
            raise RuntimeError(f"BF16 stock GDN cache dtype/layout differs: {layer.prefix}")
        pools.append(dict(layer=layer.prefix, state_dtype=str(recurrent.dtype),
                          conv_dtype=str(conv.dtype), state_shape=list(recurrent.shape),
                          state_stride=list(recurrent.stride())))
    return pools


def check_cache_initialization(worker):
    state = getattr(worker, _STATE)
    if state["cache_initialized"] or state["ready"]:
        raise RuntimeError("Live stock GDN cache reallocation is unsupported")


def initialize_cache(worker):
    check_cache_initialization(worker)
    state = getattr(worker, _STATE)
    _inspect_pools(worker)
    state["cache_initialized"] = True


def prepare_pools(worker):
    state = getattr(worker, _STATE)
    if not state["cache_initialized"] or state["ready"]:
        raise RuntimeError("Stock GDN requires a single initialized cache")
    _inspect_pools(worker)
    state["ready"] = True


def inspect_worker(worker):
    state = getattr(worker, _STATE)
    if not state["ready"]:
        raise RuntimeError("Stock GDN cache was not prepared")
    pools = _inspect_pools(worker)
    rows = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
    return dict(backend="vllm_stock", ready=True, cache_initialized=state["cache_initialized"],
                state_dtype="bfloat16", layer_count=len(pools), layers=list(state["layers"]),
                pools=pools, ordered_hooks_installed=False, ordered_scratch_bytes=0,
                profile_mode=os.environ.get("VLLM_MACH_MXFP8_MODE", "production"),
                quality_rows=int(rows) if rows else None)
