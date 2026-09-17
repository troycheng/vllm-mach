"""MoE profile upgrade and routed execution regression coverage."""

import argparse
import json
from types import SimpleNamespace

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
    with pytest.raises(ValueError, match="dense-only"):
        build_command(args, [])


def test_schedule_boundaries():
    from vllm_mach.mxfp6.moe import _qwen35_moe_schedule

    for size, expected in (
        (0, "generic"),
        (1, "small_batch"),
        (4, "small_batch"),
        (5, "grouped"),
        (96, "grouped"),
        (97, "generic"),
    ):
        assert _qwen35_moe_schedule(size) == expected


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
