# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 boundaries and changing-input replay of the TP2 MLP producer."""
import pytest
import torch

from vllm_mach.mxfp6.fused_mlp import _fused_down

def _available():
    if not torch.cuda.is_available():
        return False
    import mxfp6
    mxfp6.load_library()
    return hasattr(torch.ops.mxfp6, 'gemm_from_swiglu')


pytestmark = pytest.mark.skipif(not _available(), reason='TP2 SwiGLU extension required')


@pytest.fixture(scope='module')
def weight():
    import mxfp6
    torch.manual_seed(9181)
    return mxfp6.quantize_mxfp6(torch.randn(5120,8704,device='cuda',dtype=torch.bfloat16)*0.02)


@pytest.mark.parametrize('m',[1,2,4,8,16,24,32])
@torch.inference_mode()
def test_fused_swiglu_down_replay(m,weight):
    import mxfp6
    from vllm.model_executor.layers.activation import SiluAndMul  # load CUDA ops
    del SiluAndMul
    x=torch.randn(m,17408,device='cuda',dtype=torch.bfloat16)
    activated=torch.empty(m,8704,device='cuda',dtype=torch.bfloat16)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C.silu_and_mul(activated,x)
        mxfp6.warmup_w6a8(activated,weight,out_dtype=torch.bfloat16,iterations=1)
        _fused_down(x,weight.values.reshape(5120,6528),weight.scales)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            actual=_fused_down(x,weight.values.reshape(5120,6528),weight.scales)
        for scale in (0.0,1e-5,1.0,100.0):
            x.normal_().mul_(scale)
            graph.replay()
            torch.ops._C.silu_and_mul(activated,x)
            # Independent FP64 activation arithmetic with the explicit SiLU
            # and product BF16 boundaries used by the public vLLM CUDA op.
            gate,up=x.double().chunk(2,-1)
            denominator=(1.0+torch.exp(-gate).float()).float()
            oracle=(gate.float()/denominator).bfloat16()
            oracle=(oracle.double()*(up+0.0)).bfloat16()
            torch.testing.assert_close(activated,oracle,atol=0,rtol=0)
            refq=mxfp6.quantize_mxfp8(activated)
            values,scales=torch.ops.mxfp6.silu_and_mul_mxfp8(x)
            torch.testing.assert_close(values,refq.values,atol=0,rtol=0)
            torch.testing.assert_close(scales,refq.scales,atol=0,rtol=0)
            expected=mxfp6.gemm_from_float(activated,weight,out_dtype=torch.bfloat16)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.cuda.current_stream().wait_stream(stream)
