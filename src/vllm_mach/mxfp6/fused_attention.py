# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact TP2 decode attention sigmoid-gate/MXFP8 producer."""
from __future__ import annotations

import os
import types

import torch

from .dense import Mxfp6Sm120LinearKernel, _import_mxfp6

_REGISTERED = False
_ROWS = (1, 2, 4, 8, 16, 24, 32)


def _project(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
             scales: torch.Tensor) -> torch.Tensor:
    from mxfp6.attention import sigmoid_gate_mxfp8

    q, activation_scales = sigmoid_gate_mxfp8(x, gate)
    return torch.ops.mxfp6.gemm_w6a8_pdl(q, weight, activation_scales, scales,
        x.shape[0], weight.shape[0], 3072, 1.0, x.dtype)


def _fake(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
          scales: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _eligible(module):
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    if type(module) is not Qwen3NextAttention or not module.attn_output_gate:
        return False
    out = module.o_proj
    return (type(out) is RowParallelLinear and out.tp_size == 2
            and out.input_is_parallel and not out.reduce_results and out.bias is None
            and module.q_size == 3072 and tuple(out.weight.shape) == (5120, 2304)
            and isinstance(getattr(getattr(out, 'scheme', None), 'ocp_mx_linear', None),
                           Mxfp6Sm120LinearKernel))


def prepare(model):
    """Retain original QKV/RoPE/cache/attention and row-parallel reduction order."""
    global _REGISTERED
    mode = os.environ.get('VLLM_MACH_FUSED_ATTN_QUANT', '0')
    if mode not in ('0', '1'):
        raise ValueError('VLLM_MACH_FUSED_ATTN_QUANT must be 0 or 1')
    if mode == '0' or not any(_eligible(module) for module in model.modules()):
        return 0
    extension = _import_mxfp6()
    extension.load_library()
    try:
        from mxfp6.attention import sigmoid_gate_mxfp8  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Fused attention requires mxfp6.attention producer') from exc
    if not hasattr(torch.ops.mxfp6, 'gemm_w6a8_pdl'):
        raise RuntimeError('Fused attention requires gemm_w6a8_pdl')
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name='mach_attention_mxfp6_out', op_func=_project,
                                  mutates_args=[], fake_impl=_fake)
        _REGISTERED = True
    count = 0
    for module in model.modules():
        if getattr(module, '_mach_attention_prepared', False) or not _eligible(module):
            continue
        original = module.forward

        def forward(this, positions, hidden_states, _original=original):
            if (hidden_states.ndim != 2 or hidden_states.shape[0] not in _ROWS
                    or hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda):
                return _original(positions, hidden_states)
            qkv, _ = this.qkv_proj(hidden_states)
            q, k, v, gate = this._project_qkv_gate(qkv, positions)
            output = this.attn(q, k, v)
            return torch.ops.vllm.mach_attention_mxfp6_out(
                output, gate, this.o_proj.weight, this.o_proj.weight_scale)

        module.forward = types.MethodType(forward, module)
        module._mach_attention_prepared = True
        count += 1
    if count:
        from vllm.logger import init_logger
        init_logger('vllm.mach.attention').info('Mach prepared %d TP2 attention/MXFP8 producers', count)
    return count
