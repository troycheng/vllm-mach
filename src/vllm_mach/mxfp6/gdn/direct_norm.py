# SPDX-License-Identifier: Apache-2.0
"""Full-row gated norm for the prepared native MoE FP16 private output buffer.

Only legacy.prepare installs this path. The NEW core/state implementation stays
unchanged; its private, complete-M temporary is the existing norm kernel's out.
Unsupported execution contexts retain the original complete projection.
"""
import types


def invoke_direct_norm(layer, core, z, legacy):
    previous = getattr(legacy._CONTEXT, 'norm', None)
    legacy._CONTEXT.norm = layer.prefix
    try:
        layer._rms_norm_gated_cuda(core, z, core)
    finally:
        legacy._CONTEXT.norm = previous
    return core


def modes_allow(torch):
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    return _get_current_dispatch_mode() is None and _get_current_function_mode() is None and torch.is_inference_mode_enabled() and (not torch.is_grad_enabled()) and (not torch.compiler.is_compiling()) and (not torch.jit.is_tracing()) and (not torch.jit.is_scripting())


def hooks_empty(norm, torch):
    local = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
    global_names = ('_global_forward_pre_hooks', '_global_forward_hooks', '_global_backward_pre_hooks', '_global_backward_hooks')
    return not any((vars(norm).get(k) for k in local)) and (not any((getattr(torch.nn.modules.module, k, {}) for k in global_names))) and (getattr(norm, '_compiled_call_impl', None) is None)


def identity(method):
    return (getattr(method, '__func__', method), getattr(method, '__self__', None))


def nested_code(fn, name):
    return next((x for x in fn.__code__.co_consts if isinstance(x, types.CodeType) and x.co_name == name))


def closure_values(fn):
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))


def qualify_layer(layer, torch, NormType, legacy):
    """Idle/static native preparation identity, not model construction."""
    try:
        norm = layer.norm
        cfg = layer.vllm_config
        if not legacy._ENABLED or not legacy._INSTALLED or layer._mach_native_legacy is not True or (layer._mach_gdn_prepared is not True) or (layer.tp_size != 2) or (layer.num_k_heads != 16) or (layer.num_v_heads != 32) or (layer.get_state_dtype()[1] != torch.float16) or (layer.gdn_prefill_backend != 'triton') or (cfg.speculative_config is not None) or ('-rms_norm_gated' in cfg.compilation_config.custom_ops):
            return None
        if type(norm) is not NormType or type(norm.weight) is not torch.nn.Parameter or norm.bias is not None or (norm.weight.shape != (128,)) or (norm.weight.dtype != torch.bfloat16) or (norm.eps != layer.layer_norm_epsilon) or (norm.group_size is not None) or (norm.norm_before_gate is not True) or (norm.activation not in ('silu', 'swish')) or (not hooks_empty(norm, torch)):
            return None
        forward = norm._forward_method
        direct = layer._rms_norm_gated_cuda
        if type(forward) is not types.FunctionType or type(direct) is not types.FunctionType or forward.__code__ is not nested_code(legacy.prepare, 'forward') or (direct.__code__ is not nested_code(legacy.prepare, 'decode_norm')) or (forward.__globals__ is not vars(legacy)) or (direct.__globals__ is not vars(legacy)):
            return None
        f = closure_values(forward)
        d = closure_values(direct)
        if f.get('layer') is not layer or f.get('norm') is not norm or d.get('layer') is not layer:
            return None
        if identity(f.get('original')) != identity(norm.forward_native):
            return None
        original = d.get('original_decode')
        if getattr(original, '__self__', None) is not layer:
            return None
        return dict(layer=layer, norm=norm, norm_type=NormType, forward=forward, direct=direct, weight=norm.weight, weight_type=type(norm.weight), weight_shape=tuple(norm.weight.shape), weight_dtype=norm.weight.dtype, weight_device=norm.weight.device, scalar=(norm.eps, norm.group_size, norm.norm_before_gate, norm.activation), norm_forward=identity(norm.forward), prefix=layer.prefix, config=cfg)
    except (AttributeError, StopIteration, TypeError):
        return None


def tensor_guard(core, z, torch, guard):
    if type(core) is not torch.Tensor or type(z) is not torch.Tensor:
        return False
    if core.device.type != 'cuda' or z.device != core.device:
        return False
    if core.dtype != torch.bfloat16 or z.dtype != core.dtype:
        return False
    if core.ndim != 3 or core.shape[0] <= 0 or tuple(core.shape[1:]) != (16, 128) or (z.shape != core.shape):
        return False
    if core.stride(-1) != 1 or z.stride(-1) != 1 or (not core.is_contiguous()):
        return False
    if torch.cuda.current_device() != core.device.index:
        return False
    return guard.calc_rows_per_block(core.numel() // 128, core.device) in (1, 2, 4)


def eligible(layer, core, z, bound, torch, legacy, guard):
    if not modes_allow(torch):
        return False
    if bound is None or layer is not bound['layer'] or layer.norm is not bound['norm']:
        return False
    norm = layer.norm
    if not legacy._ENABLED or not legacy._INSTALLED or (not hooks_empty(norm, torch)) or (norm._forward_method is not bound['forward']) or (layer._rms_norm_gated_cuda is not bound['direct']) or (identity(norm.forward) != bound['norm_forward']) or (layer.prefix != bound['prefix']) or (layer.vllm_config is not bound['config']) or (layer.vllm_config.speculative_config is not None) or (layer.get_state_dtype()[1] != torch.float16) or (norm.bias is not None) or ((norm.eps, norm.group_size, norm.norm_before_gate, norm.activation) != bound['scalar']) or (norm.weight is not bound['weight']) or (type(norm.weight) is not bound['weight_type']):
        return False
    if tuple(norm.weight.shape) != bound['weight_shape'] or norm.weight.dtype != bound['weight_dtype'] or norm.weight.device != bound['weight_device']:
        return False
    return tensor_guard(core, z, torch, guard) and norm.weight.device == core.device


def _bind_projection(layer, original_bound, bound, torch, legacy, guard):
    # Keep the qualified candidate's admission/direct closure boundaries.
    admission = lambda current, core, z: eligible(
        current, core, z, bound, torch, legacy, guard)
    direct = lambda current, core, z: invoke_direct_norm(
        current, core, z, legacy)

    def output_projection(self, core_attn_out, z):
        if not admission(self, core_attn_out, z):
            return original_bound(core_attn_out, z)
        core_attn_out = direct(self, core_attn_out, z)
        output, _ = self.out_proj(core_attn_out.flatten(-2))
        return output

    return types.MethodType(output_projection, layer)


def prepare(layer):
    """Bind after the native norm closures, before any CUDA graph capture."""
    import torch
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard
    from . import legacy

    original = layer._output_projection
    if (getattr(original, '__self__', None) is not layer
            or getattr(original, '__func__', None)
            is not QwenGatedDeltaNetAttention._output_projection):
        return
    bound = qualify_layer(layer, torch, RMSNormGated, legacy)
    if bound is None:
        return
    layer._output_projection = _bind_projection(
        layer, original, bound, torch, legacy, layernorm_guard)
