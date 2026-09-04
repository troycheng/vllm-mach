from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_mach.exl3 import fused_allreduce
from vllm_mach.exl3 import mxfp6_hybrid as hybrid


def test_fused_allreduce_falls_back_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv(fused_allreduce.FUSED_AR_ENV, raising=False)
    expected = (object(), object())
    monkeypatch.setattr(fused_allreduce, "_fallback", lambda *_args: expected)
    hidden = torch.empty((4, 8), dtype=torch.bfloat16)

    assert fused_allreduce.fused_allreduce_gemma_rms_norm_mxfp8(
        hidden, hidden.clone(), SimpleNamespace()
    ) is expected


def test_fused_allreduce_rejects_unpatched_flashinfer_header(
    monkeypatch, tmp_path
) -> None:
    package = tmp_path / "flashinfer"
    header = (
        package / "data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh"
    )
    header.parent.mkdir(parents=True)
    header.write_text("unpatched", encoding="utf-8")
    module = SimpleNamespace(__file__=str(package / "__init__.py"))

    with pytest.raises(RuntimeError, match="requires the vLLM Mach FlashInfer profile"):
        fused_allreduce._verify_flashinfer_header(module)


def test_fused_allreduce_writes_mxfp6_packed_layout(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    monkeypatch.setenv(fused_allreduce.FUSED_AR_ENV, "1")
    monkeypatch.setattr(fused_allreduce, "_eligible", lambda *_args: True)
    calls = {}

    class _Workspace:
        backend = "trtllm"

        @staticmethod
        def is_buffer_size_sufficient(**_kwargs):
            return True

    class _Packed:
        def __init__(self, values, scales, rows, k):
            self.values = values
            self.scales = scales
            self.rows = rows
            self.k = k

    class _Patterns:
        kARResidualRMSNormPerTokenGroupFP8PackedQuant = 8

    class _Layouts:
        SWIZZLED_128x4 = 0

    def allreduce_fusion(**kwargs):
        calls.update(kwargs)
        kwargs["residual_out"].copy_(kwargs["input"] + kwargs["residual_in"])

    dependencies = SimpleNamespace(
        comm=SimpleNamespace(
            AllReduceFusionPattern=_Patterns,
            QuantizationSFLayout=_Layouts,
            allreduce_fusion=allreduce_fusion,
        ),
        runtime=SimpleNamespace(MXFP8Tensor=_Packed),
        get_workspace=lambda **_kwargs: _Workspace(),
        get_rank=lambda: 0,
        get_world_size=lambda: 2,
        get_tp_group=lambda: SimpleNamespace(cpu_group=object()),
    )
    monkeypatch.setattr(fused_allreduce, "_load_dependencies", lambda: dependencies)
    hidden = torch.ones((4, 5120), dtype=torch.bfloat16)
    residual = torch.full_like(hidden, 2)
    norm = SimpleNamespace(
        weight=torch.ones(5120, dtype=torch.bfloat16),
        variance_epsilon=1e-6,
    )

    activation, residual_out = (
        fused_allreduce.fused_allreduce_gemma_rms_norm_mxfp8(
            hidden, residual, norm
        )
    )

    assert isinstance(activation, _Packed)
    assert (activation.rows, activation.k) == (4, 5120)
    assert activation.values.dtype == torch.uint8
    assert activation.scales.numel() == 128 * 160
    assert calls["pattern"] == 8
    assert calls["layout_code"] == 0
    assert calls["scale_out"].shape == (4, 40)
    assert calls["scale_out"].stride() == (1, 4)
    assert calls["block_quant_group_size"] == 32
    assert calls["weight_bias"] == 1.0
    assert torch.all(residual_out == 3)
