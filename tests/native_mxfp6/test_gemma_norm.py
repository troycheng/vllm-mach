"""Numerical and CUDA graph coverage for the TP1 Gemma norm producer."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n", [2048, 2560, 5120])
@pytest.mark.parametrize("m", [0, 1, 16, 32, 513])
@pytest.mark.parametrize("with_residual", [False, True])
def test_norm(n, m, with_residual):
    from vllm_mach.mxfp6.gemma_norm import _impl

    torch.manual_seed(17)
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    before = x.clone()
    y, summed = _impl(x, residual, weight, 1e-6)
    value = x.float() if residual is None else x.float() + residual.float()
    ref = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
           * (weight.float() + 1)).bfloat16()
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=8e-3)
    assert torch.equal(x, before)
    if residual is not None:
        assert torch.equal(summed, value.bfloat16())


def test_graph_uses_changed_input_and_weights():
    from vllm_mach.mxfp6.gemma_norm import _impl

    x = torch.randn(16, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    _impl(x, residual, weight, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, summed = torch.ops.vllm.mach_gemma_norm(x, residual, weight, 1e-6)
    for scale in (0., 1., 10.):
        x.normal_().mul_(scale)
        residual.normal_().mul_(scale)
        weight.normal_()
        graph.replay()
        value = x.float() + residual.float()
        ref = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
               * (weight.float() + 1)).bfloat16()
        torch.testing.assert_close(y, ref, atol=1e-3, rtol=8e-3)
        assert torch.equal(summed, value.bfloat16())
