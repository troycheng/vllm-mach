"""MoE profile upgrade and routed execution regression coverage."""

import argparse
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_mach.mxfp6.serve import build_command


def test_moe_launcher_enables_gdn_ar_norm_and_optional_head(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    args = argparse.Namespace(
        model=tmp_path,
        fp16_ssm=False,
        lossless_prefill=False,
        owner_prefill=False,
        nvfp4_lm_head=False,
        verify_prefill=False,
    )
    command, env = build_command(args, [])
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--compilation-config" not in command
    assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
    assert env["VLLM_QWEN3_5_FUSED_AR_NORM"] == "1"
    assert env["VLLM_MACH_GDN_PERSISTENT"] == "1"
    assert env["VLLM_MACH_GDN_BA_OVERLAP"] == "1"
    args.gdn_persistent = args.gdn_ba_overlap = False
    _, env = build_command(args, [])
    assert env["VLLM_MACH_GDN_PERSISTENT"] == "0"
    assert env["VLLM_MACH_GDN_BA_OVERLAP"] == "0"
    args.nvfp4_lm_head = True
    args.fused_ar_norm = False
    _, env = build_command(args, [])
    assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == "1"
    assert env["VLLM_QWEN3_5_FUSED_AR_NORM"] == "0"
    args.fp16_ssm = True
    command, env = build_command(args, [])
    assert command[command.index("--mamba-ssm-cache-dtype") + 1] == "float16"
    assert env["VLLM_QWEN3_5_FP16_SSM"] == "1"
    args.lossless_prefill = args.owner_prefill = None
    _, env = build_command(args, [])
    assert env["VLLM_SM120_LOSSLESS_PREFILL"] == "0"
    assert env["VLLM_SM120_OWNER_PREFILL"] == "0"
    for flag in ("lossless_prefill", "owner_prefill"):
        setattr(args, flag, True)
        with pytest.raises(ValueError, match="dense-only"):
            build_command(args, [])
        setattr(args, flag, None)


@pytest.mark.parametrize(
    "size, default, enabled",
    [
        (0, "generic", "generic"),
        (1, "small_batch", "small_batch"),
        (4, "small_batch", "small_batch"),
        (5, "grouped", "grouped"),
        (96, "grouped", "grouped"),
        (97, "generic", "compact"),
        (160, "generic", "compact"),
        (161, "generic", "generic"),
    ],
)
def test_schedule_boundaries(size, default, enabled):
    from vllm_mach.mxfp6.moe import _qwen35_moe_schedule

    assert _qwen35_moe_schedule(size) == default
    assert _qwen35_moe_schedule(size, high_m_compact=True) == enabled


def test_compact_moe_graph_cache_mode(monkeypatch):
    from vllm_mach.mxfp6 import moe, moe_ar_norm
    from vllm_mach.stack import CACHE_NAMESPACE

    monkeypatch.setattr(moe_ar_norm, "prepare_compiled_ar_norm", lambda config: None)
    config = SimpleNamespace(additional_config={})
    moe._prepare_qwen35_moe_graph_cache(config, high_m_compact=False)
    original = config.additional_config["mxfp6_sm120_moe_graph"]
    assert original == "mach_qwen35_tp2_v2_" + CACHE_NAMESPACE
    moe._prepare_qwen35_moe_graph_cache(config, high_m_compact=True)
    assert config.additional_config["mxfp6_sm120_moe_graph"] != original


def test_compact_moe_rounds_routed_before_shared_add(monkeypatch):
    from vllm_mach.mxfp6 import moe

    hidden = torch.empty(97, 2048, dtype=torch.bfloat16)
    logits = torch.empty(97, 256, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    shared_output = torch.full_like(hidden, 2**-8)
    shared = Mock(output=shared_output)
    shared.maybe_forward_async.return_value = True
    runner = object.__new__(moe.Mxfp6Sm120MoERunner)
    packed = torch.empty(257)
    runner.__dict__.update(
        _mxfp6_sm120_qwen35_small_batch_state=SimpleNamespace(
            high_m_compact=True, w1=packed, w1_scales=packed, w2=packed, w2_scales=packed
        ),
        _shared_experts=shared,
        routed_experts=Mock(),
        gate=Mock(return_value=(logits, None)),
        _sequence_parallel_context=nullcontext,
        _maybe_dispatch=lambda hidden, logits: (hidden, logits),
    )
    workspace = object()
    monkeypatch.setattr(moe, "_qwen35_grouped_workspace", lambda _: (workspace, output))

    def reduce_routed(workspace_arg, output_arg):
        assert workspace_arg is workspace and output_arg is output
        shared.wait.assert_called_once_with()
        # The two BF16 rounding steps give 1. Fusing the shared add gives 1 + 2**-7.
        output_arg.fill_(1 + 2**-8)

    native = SimpleNamespace(
        qwen35_compact_gemm_out=Mock(), qwen35_grouped_reduce_out=reduce_routed
    )
    monkeypatch.setitem(sys.modules, "mxfp6", native)
    assert runner._forward_impl(hidden, logits, hidden) is output
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
    args, kwargs = native.qwen35_compact_gemm_out.call_args
    assert not kwargs
    assert args[0] is workspace and args[1] is hidden and args[2] is logits
    assert [tuple(tensor.shape) for tensor in args[3:]] == [(256,)] * 4


@pytest.mark.parametrize("rows", [5, 32, 96])
def test_grouped_output_survives_workspace_reuse(monkeypatch, rows):
    """A later invocation must not overwrite an earlier custom-op result."""
    mxfp6 = pytest.importorskip("mxfp6")
    from vllm_mach.mxfp6 import moe

    shapes = mxfp6.qwen35_grouped_workspace_shapes(rows)
    scratch = [torch.empty(shape, dtype=torch.bfloat16) for shape in shapes]
    manager = SimpleNamespace(get_simultaneous=lambda *args: scratch)
    monkeypatch.setattr(moe, "current_workspace_manager", lambda: manager)
    hidden = torch.empty(rows, 2048, dtype=torch.bfloat16)
    _, first = moe._qwen35_grouped_workspace(hidden)
    first.fill_(3)
    _, second = moe._qwen35_grouped_workspace(hidden)
    second.fill_(7)
    for buffer in scratch:
        buffer.zero_()
    torch.testing.assert_close(first, torch.full_like(first, 3))
    torch.testing.assert_close(second, torch.full_like(second, 7))


@torch.inference_mode()
def test_routed_experts_changing_graph_inputs():
    """Compare the complete adapter against independent per-expert dense GEMMs."""
    from vllm_mach.mxfp6.moe_utils import is_mxfp6_sm120_moe_available

    if not is_mxfp6_sm120_moe_available():
        pytest.skip("requires native MXFP6 MoE on SM120")
    import mxfp6
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_mach.mxfp6.moe import Mxfp6Sm120Experts

    torch.manual_seed(12035)
    e, h, i, m, topk = 4, 256, 128, 7, 2
    p1 = mxfp6.quantize_mxfp6(
        torch.randn(e * 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    p2 = mxfp6.quantize_mxfp6(
        torch.randn(e * h, i, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    w1, w2 = p1.values.view(e, 2 * i, h * 3 // 4), p2.values.view(e, h, i * 3 // 4)
    s1, s2 = p1.scales.view(e, -1), p2.scales.view(e, -1)
    expert = SimpleNamespace(w1_scale=s1, w2_scale=s2)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]] * m, device="cuda", dtype=torch.int32)
    weights = torch.rand(m, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    out = torch.empty_like(x)
    ws1 = torch.empty(m * topk, max(2 * i, h), device="cuda", dtype=x.dtype)
    ws2 = torch.empty(m * topk, max(i, h), device="cuda", dtype=x.dtype)

    def run():
        Mxfp6Sm120Experts.apply(
            expert,
            out,
            x,
            w1,
            w2,
            weights,
            ids,
            MoEActivation.SILU,
            e,
            None,
            None,
            None,
            ws1,
            ws2,
            None,
            False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shift in range(3):
        x.normal_()
        ids.copy_(
            torch.tensor(
                [[(j + shift) % e, (j + shift + 1) % e] for j in range(m)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        graph.replay()
        refs = []
        for k in range(e):
            a = mxfp6.gemm_from_float(
                x, mxfp6.PackedMXFP6Tensor(w1[k], s1[k], 2 * i, h), out_dtype=x.dtype
            )
            # Match the producer contract: round SiLU to BF16 before multiplying.
            gate, up = a.float().chunk(2, dim=-1)
            a = (torch.nn.functional.silu(gate).to(x.dtype).float() * up).to(x.dtype)
            refs.append(
                mxfp6.gemm_from_float(
                    a, mxfp6.PackedMXFP6Tensor(w2[k], s2[k], h, i), out_dtype=x.dtype
                )
            )
        ref = sum(
            torch.stack(refs)[ids[:, k].long(), torch.arange(m, device="cuda")].float()
            * weights[:, k, None]
            for k in range(topk)
        ).to(x.dtype)
        torch.testing.assert_close(out, ref, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_moe_preserves_explicit_graph_sizes_and_vllm_defaults(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    args = argparse.Namespace(
        model=tmp_path,
        fp16_ssm=False,
        lossless_prefill=False,
        owner_prefill=False,
        nvfp4_lm_head=True,
        verify_prefill=False,
    )
    sizes = [
        "1",
        "2",
        "4",
        "8",
        "10",
        "12",
        "14",
        "16",
        "20",
        "24",
        "28",
        "32",
        "36",
        "40",
        "48",
        "56",
        "64",
        "72",
        "80",
        "96",
        "112",
        "128",
        "160",
        "192",
        "224",
        "256",
        "320",
        "384",
        "448",
        "512",
        "640",
        "768",
        "784",
        "896",
        "1024",
        "1280",
        "1536",
        "1792",
        "2048",
    ]
    extra = ["--max-num-seqs", "64", "--cudagraph-capture-sizes", *sizes]
    command, _ = build_command(args, extra)
    assert command[-len(extra) :] == extra
    assert "--compilation-config" not in command
    assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
    # Explicit overrides still work, including reproducing the old profile.
    override = [
        "--compilation-config",
        '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    command, _ = build_command(args, extra + override)
    assert command.count("--compilation-config") == 1
    assert command[-len(override) :] == override
