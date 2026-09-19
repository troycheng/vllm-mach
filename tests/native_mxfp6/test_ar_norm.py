"""CPU guard and handoff tests for the Dense AR/GemmaRMSNorm/MXFP8 path."""

from __future__ import annotations

import sys
import types

import pytest
import torch


def _metadata(rows: int, **changes):
    return types.SimpleNamespace(
        **dict(
            dict(
                num_prefills=0,
                num_spec_decodes=0,
                spec_sequence_masks=None,
                num_decodes=rows,
                num_actual_tokens=rows,
            ),
            **changes,
        )
    )


class _Input:
    """Tensor-shaped CUDA stand-in so guard tests remain CPU-only."""

    def __init__(self, rows=2, *, contiguous=True):
        self.ndim = 2
        self.shape = (rows, 5120)
        self.dtype = torch.bfloat16
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def _forward_context(monkeypatch, metadata):
    import vllm.forward_context

    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"dense": metadata}),
    )


@pytest.mark.parametrize(
    "rows, change",
    [
        (1, {}),
        (33, {}),
        (2, {"num_prefills": 1}),
        (2, {"num_spec_decodes": 1}),
        (2, {"spec_sequence_masks": object()}),
        (2, {"num_actual_tokens": 1}),
    ],
)
def test_decode_guard_rejects_non_decode_boundaries(monkeypatch, rows, change):
    from vllm_mach.mxfp6 import ar_norm

    _forward_context(monkeypatch, _metadata(rows, **change))
    assert not ar_norm.decode_supported(_Input(rows))


def test_decode_guard_requires_valid_mapping_and_contiguous_bf16(monkeypatch):
    from vllm_mach.mxfp6 import ar_norm

    _forward_context(monkeypatch, _metadata(2))
    assert not ar_norm.decode_supported(_Input(2, contiguous=False))
    import vllm.forward_context

    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=object()),
    )
    assert not ar_norm.decode_supported(_Input(2))
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"dense": object()}),
    )
    assert not ar_norm.decode_supported(_Input(2))
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            attn_metadata={"dense": types.SimpleNamespace(num_spec_decodes=0)}
        ),
    )
    assert not ar_norm.decode_supported(_Input(2))


def test_prepare_disabled_and_no_eligible_never_load_native(monkeypatch):
    from vllm_mach.mxfp6 import ar_norm

    model = torch.nn.Sequential(torch.nn.Identity())
    monkeypatch.setattr(ar_norm, "load_library", lambda: pytest.fail("native loaded"))
    monkeypatch.setenv("VLLM_MACH_FUSED_AR_QUANT", "0")
    assert ar_norm.prepare(model) == 0
    monkeypatch.setenv("VLLM_MACH_FUSED_AR_QUANT", "auto")
    monkeypatch.setattr(ar_norm, "_eligible", lambda _: False)
    assert ar_norm.prepare(model) == 0


def test_eligibility_is_exact_dense_bf16_sm120_tp2_with_prepared_producers(monkeypatch):
    from vllm.model_executor.layers import layernorm
    from vllm.model_executor.models import qwen3_5
    from vllm_mach.mxfp6 import ar_norm
    from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel

    class Norm:
        pass

    class Decoder:
        pass

    class Weight:
        shape = (5120,)
        dtype = torch.bfloat16
        is_cuda = True
        is_contiguous = staticmethod(lambda: True)
        device = object()

    monkeypatch.setattr(layernorm, "GemmaRMSNorm", Norm)
    monkeypatch.setattr(qwen3_5, "Qwen3_5DecoderLayer", Decoder)
    monkeypatch.setattr(ar_norm.torch.cuda, "get_device_capability", lambda _: (12, 0))
    kernel = object.__new__(Mxfp6Sm120LinearKernel)
    projection = types.SimpleNamespace(
        tp_size=2,
        bias=None,
        weight=types.SimpleNamespace(shape=(8192, 3840)),
        scheme=types.SimpleNamespace(ocp_mx_linear=kernel),
    )
    layer = Decoder()
    vars(layer).update(
        use_fused_ar_gemma_norm=True,
        layer_scale=False,
        mlp=types.SimpleNamespace(_mach_swiglu_prepared=True),
        layer_type="linear_attention",
        linear_attn=types.SimpleNamespace(
            _mach_gdn_prepared=True, in_proj_qkvz=projection
        ),
    )
    # Construct norms with the exact runtime type; subclasses are deliberately excluded.
    layer.input_layernorm = Norm()
    layer.input_layernorm.weight = Weight()
    layer.post_attention_layernorm = Norm()
    layer.post_attention_layernorm.weight = Weight()
    assert ar_norm._eligible(layer)
    layer.linear_attn._mach_gdn_prepared = False
    assert not ar_norm._eligible(layer)
    layer.linear_attn._mach_gdn_prepared = True
    layer.input_layernorm = type("DerivedNorm", (Norm,), {})()
    layer.input_layernorm.weight = Weight()
    assert not ar_norm._eligible(layer)


def test_loader_reports_missing_wheel_and_abi_mismatch(monkeypatch):
    from vllm_mach.mxfp6 import ar_norm

    ar_norm._loaded = False
    monkeypatch.setattr(
        ar_norm.metadata,
        "version",
        lambda package: (_ for _ in ()).throw(ar_norm.metadata.PackageNotFoundError()),
    )
    with pytest.raises(RuntimeError, match=r"flashinfer-python==0.6.18"):
        ar_norm.load_library()

    versions = {
        "flashinfer-python": "0.6.18+local",
        "mxfp6-sm120": "0.2.1",
        "vllm-mach-ar-norm": "0.1.0a1",
    }
    monkeypatch.setattr(ar_norm.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(ar_norm.util, "find_spec", lambda _: types.SimpleNamespace(origin="/tmp/x.so"))
    monkeypatch.setattr(ar_norm.torch.ops, "load_library", lambda _: None)
    monkeypatch.setattr(
        ar_norm.torch.ops,
        "mach_norm_quant",
        types.SimpleNamespace(abi=lambda: "wrong", run=object()),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="native ABI mismatch"):
        ar_norm.load_library()


def test_prepare_is_idempotent_and_preserves_original_forward(monkeypatch):
    from vllm_mach.mxfp6 import ar_norm

    class Layer:
        layer_type = "full_attention"

        def forward(self, hidden_states, residual=None, positions=None, **kwargs):
            return ("original", hidden_states, residual, positions, kwargs)

    layer = Layer()
    model = types.SimpleNamespace(modules=lambda: [layer])
    loads = []
    monkeypatch.setenv("VLLM_MACH_FUSED_AR_QUANT", "auto")
    monkeypatch.setattr(ar_norm, "_eligible", lambda candidate: candidate is layer)
    monkeypatch.setattr(ar_norm, "load_library", lambda: loads.append("load"))
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _: types.SimpleNamespace(info=lambda *args: None)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    assert ar_norm.prepare(model) == 1
    assert ar_norm.prepare(model) == 0
    assert loads == ["load"]

    monkeypatch.setattr(ar_norm, "decode_supported", lambda *_: False)
    result = layer.forward("h", "r", "p", marker="kept")
    assert result == ("original", "h", "r", "p", {"marker": "kept"})


def test_workspace_rejection_uses_original_keyword_contract(monkeypatch):
    from vllm_mach.mxfp6 import ar_norm

    seen = {}

    def original(**kwargs):
        seen.update(kwargs)
        return "baseline"

    hidden = torch.zeros(2, 5120, dtype=torch.bfloat16)
    residual = torch.zeros_like(hidden)
    layer = types.SimpleNamespace(layer_type="full_attention")
    monkeypatch.setattr(ar_norm, "decode_supported", lambda value, prefix=None: value is hidden)
    monkeypatch.setattr(ar_norm, "_workspace", lambda _: None)
    assert (
        ar_norm._forward(
            layer, hidden, residual, "positions", original=original, flag=3
        )
        == "baseline"
    )
    assert seen["hidden_states"] is hidden
    assert seen["residual"] is residual
    assert seen["positions"] == "positions"
    assert seen["flag"] == 3


def test_gdn_quantized_handoff_uses_projected_qkv_not_original(monkeypatch):
    """The GDN consumer must use the explicit codes/scales producer contract."""
    import vllm.forward_context
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm_mach.mxfp6 import ar_norm, gdn_decode
    from vllm_mach.mxfp6.gdn import persistent

    rows, heads, hidden, qkv_dim = 2, 24, 5120, 5120
    indices = torch.arange(rows, dtype=torch.int32)
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"dense": _metadata(rows, non_spec_state_indices_tensor=indices)}),
    )
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(persistent, "execute", lambda *args: args[-1].zero_())
    calls = []
    mixed = torch.zeros(rows, qkv_dim + heads * 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        ar_norm,
        "projected",
        lambda codes, scales, projection: calls.append((codes, scales, projection)) or mixed,
    )

    def norm(core, gate, out):
        out.copy_(core)

    projection = object()
    layer = types.SimpleNamespace(
        num_k_heads=16,
        num_v_heads=48,
        in_proj_ba=types.SimpleNamespace(weight=torch.zeros(48, hidden, dtype=torch.bfloat16)),
        in_proj_qkvz=projection,
        conv1d=types.SimpleNamespace(weight=torch.zeros(qkv_dim, 1, 4, dtype=torch.bfloat16), bias=None),
        _mach_gdn_ba=torch.zeros(hidden, 48, dtype=torch.bfloat16),
        _mach_gdn_bias=torch.zeros(qkv_dim, dtype=torch.bfloat16),
        A_log=torch.zeros(heads), dt_bias=torch.zeros(heads, dtype=torch.bfloat16),
        prefix="dense",
        kv_cache=(torch.zeros(rows + 1, qkv_dim, 3, dtype=torch.bfloat16), torch.zeros(rows + 1, heads, 128, 128)),
        _mach_gdn_output_fused=False,
        _rms_norm_gated_cuda=norm,
        out_proj=lambda value: (value, None),
    )
    codes, scales = object(), object()
    output = gdn_decode._forward(layer, lambda _: pytest.fail("baseline qkv used"), True, None,
                                 torch.zeros(rows, hidden, dtype=torch.bfloat16), quantized=(codes, scales))
    assert output.shape == (rows, heads * 128)
    assert calls == [(codes, scales, projection)]
