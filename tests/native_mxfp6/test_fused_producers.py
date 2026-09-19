"""CPU-only guards for capability-gated Dense producer installation."""

from __future__ import annotations

import sys
import types

import pytest
import torch


def _install_mlp_type_modules(monkeypatch):
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    qwen = types.ModuleType("vllm.model_executor.models.qwen2_moe")
    activation = types.ModuleType("vllm.model_executor.layers.activation")
    linear.MergedColumnParallelLinear = type("MergedColumnParallelLinear", (), {})
    linear.RowParallelLinear = type("RowParallelLinear", (), {})
    qwen.Qwen2MoeMLP = type("Qwen2MoeMLP", (), {})
    activation.SiluAndMul = type("SiluAndMul", (), {})
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    monkeypatch.setitem(sys.modules, qwen.__name__, qwen)
    monkeypatch.setitem(sys.modules, activation.__name__, activation)
    return linear, qwen, activation


def _dense_mlp(monkeypatch, *, bias=False, partial=True, expert=False):
    from vllm_mach.mxfp6 import fused_mlp
    from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel

    linear, qwen, activation = _install_mlp_type_modules(monkeypatch)
    native = object.__new__(Mxfp6Sm120LinearKernel)

    def layer(cls, shape, **attrs):
        value = cls()
        value.weight = torch.empty(shape, device="meta")
        value.bias = object() if bias else None
        value.scheme = types.SimpleNamespace(ocp_mx_linear=native)
        vars(value).update(attrs)
        return value

    module = qwen.Qwen2MoeMLP()
    module.expert_gate = object() if expert else None
    module.act_fn = activation.SiluAndMul()
    module.gate_up_proj = layer(
        linear.MergedColumnParallelLinear, (17408, 3840), tp_size=2
    )
    module.down_proj = layer(
        linear.RowParallelLinear,
        (5120, 6528),
        tp_size=2,
        input_is_parallel=True,
        reduce_results=not partial,
    )
    return module


def test_swiglu_eligibility_is_dense_tp2_partial_output_only(monkeypatch):
    from vllm_mach.mxfp6.fused_mlp import _eligible

    assert _eligible(_dense_mlp(monkeypatch))
    assert not _eligible(_dense_mlp(monkeypatch, bias=True))
    assert not _eligible(_dense_mlp(monkeypatch, partial=False))
    assert not _eligible(_dense_mlp(monkeypatch, expert=True))


@pytest.mark.parametrize("mode", ["auto", "1"])
def test_swiglu_missing_operator_raises_for_enabled_mode(monkeypatch, mode):
    from vllm_mach.mxfp6 import fused_mlp

    model = torch.nn.Sequential(torch.nn.Identity())
    monkeypatch.setattr(fused_mlp, "_eligible", lambda _: True)
    monkeypatch.setattr(
        fused_mlp, "_import_mxfp6", lambda: types.SimpleNamespace(load_library=lambda: None)
    )
    monkeypatch.setattr(fused_mlp.torch.ops, "mxfp6", types.SimpleNamespace(), raising=False)
    monkeypatch.setenv("VLLM_MACH_FUSED_SWIGLU_QUANT", mode)
    with pytest.raises(RuntimeError, match="gemm_from_swiglu"):
        fused_mlp.prepare(model)


def test_swiglu_zero_does_not_probe_extension(monkeypatch):
    from vllm_mach.mxfp6 import fused_mlp

    monkeypatch.setenv("VLLM_MACH_FUSED_SWIGLU_QUANT", "0")
    monkeypatch.setattr(
        fused_mlp, "_import_mxfp6", lambda: pytest.fail("disabled path loaded extension")
    )
    assert fused_mlp.prepare(torch.nn.Identity()) == 0


@pytest.mark.parametrize("mode", ["auto", "1"])
def test_gdn_missing_operator_raises_for_enabled_mode(monkeypatch, mode):
    from vllm_mach.mxfp6 import gdn_output

    layer = object()
    monkeypatch.setattr(gdn_output, "_eligible", lambda candidate: candidate is layer)
    monkeypatch.setattr(
        gdn_output, "_import_mxfp6", lambda: types.SimpleNamespace(load_library=lambda: None)
    )
    monkeypatch.setattr(gdn_output.torch.ops, "mxfp6", types.SimpleNamespace(), raising=False)
    monkeypatch.setenv("VLLM_MACH_FUSED_GDN_QUANT", mode)
    with pytest.raises(RuntimeError, match="gemm_from_gdn"):
        gdn_output.prepare(layer)


def test_gdn_zero_does_not_probe_extension(monkeypatch):
    from vllm_mach.mxfp6 import gdn_output

    monkeypatch.setenv("VLLM_MACH_FUSED_GDN_QUANT", "0")
    monkeypatch.setattr(
        gdn_output, "_import_mxfp6", lambda: pytest.fail("disabled path loaded extension")
    )
    assert not gdn_output.prepare(object())


@pytest.mark.parametrize("rows", [3, 4096])
def test_swiglu_prefill_and_unsupported_rows_keep_original(monkeypatch, rows):
    from vllm_mach.mxfp6 import fused_mlp

    class MLP(torch.nn.Module):
        def forward(self, value):
            return ("original", value)

    class Input:
        ndim = 2
        shape = (rows, 5120)
        dtype = torch.bfloat16
        is_cuda = True

    model = torch.nn.Sequential(MLP())
    mlp = model[0]
    monkeypatch.setenv("VLLM_MACH_FUSED_SWIGLU_QUANT", "auto")
    monkeypatch.setattr(fused_mlp, "_eligible", lambda module: module is mlp)
    monkeypatch.setattr(
        fused_mlp, "_import_mxfp6", lambda: types.SimpleNamespace(load_library=lambda: None)
    )
    monkeypatch.setattr(
        fused_mlp.torch.ops,
        "mxfp6",
        types.SimpleNamespace(gemm_from_swiglu=object()),
        raising=False,
    )
    monkeypatch.setattr(fused_mlp, "_register", lambda: None)

    assert fused_mlp.prepare(model) == 1
    result = mlp(Input())
    assert result[0] == "original"
    assert isinstance(result[1], Input)


def test_gdn_moe_head_geometry_never_calls_dense_output_producer(monkeypatch):
    """The 16-head geometry must retain its existing heads-based norm path."""
    import vllm.forward_context
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm_mach.mxfp6 import gdn_decode, gdn_output
    from vllm_mach.mxfp6.gdn import persistent

    rows, heads, hidden, qkv_dim = 2, 16, 2048, 4096
    indices = torch.arange(rows, dtype=torch.int32)
    metadata = types.SimpleNamespace(
        spec_sequence_masks=None,
        num_spec_decodes=0,
        num_prefills=0,
        num_decodes=rows,
        num_actual_tokens=rows,
        non_spec_state_indices_tensor=indices,
    )
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"test": metadata}),
    )
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(persistent, "execute", lambda *args: args[-1].zero_())
    monkeypatch.setattr(
        gdn_output,
        "project",
        lambda *args: pytest.fail("16-head MoE reached Dense output producer"),
    )
    norm_shapes = []

    def norm(core, gate, out):
        norm_shapes.append(gate.shape)
        out.copy_(core)

    layer = types.SimpleNamespace(
        num_k_heads=16,
        num_v_heads=32,
        in_proj_ba=types.SimpleNamespace(
            weight=torch.zeros(2 * heads, hidden, dtype=torch.bfloat16)
        ),
        in_proj_qkvz=lambda _: (
            torch.zeros(rows, qkv_dim + heads * 128, dtype=torch.bfloat16),
            None,
        ),
        conv1d=types.SimpleNamespace(
            weight=torch.zeros(qkv_dim, 1, 4, dtype=torch.bfloat16), bias=None
        ),
        _mach_gdn_ba=torch.zeros(hidden, 2 * heads, dtype=torch.bfloat16),
        _mach_gdn_bias=torch.zeros(qkv_dim, dtype=torch.bfloat16),
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads, dtype=torch.bfloat16),
        activation="silu",
        prefix="test",
        kv_cache=(
            torch.zeros(rows + 1, qkv_dim, 3, dtype=torch.bfloat16),
            torch.zeros(rows + 1, heads, 128, 128),
        ),
        _mach_gdn_output_fused=False,
        _rms_norm_gated_cuda=norm,
        out_proj=lambda value: (value, None),
    )
    output = gdn_decode._forward(
        layer,
        lambda _: pytest.fail("unexpected baseline fallback"),
        True,
        None,
        torch.zeros(rows, hidden, dtype=torch.bfloat16),
    )
    assert output.shape == (rows, heads * 128)
    assert norm_shapes == [(rows, heads, 128)]
