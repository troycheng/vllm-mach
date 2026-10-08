# SPDX-License-Identifier: Apache-2.0
"""SwiGLU producer precision and changing-input compiled CUDA graph replay."""
import pytest
import torch
from vllm_mach.mxfp6.dense_mxfp8 import Mxfp8Sm120LinearKernel

pytestmark = pytest.mark.skipif(
    not Mxfp8Sm120LinearKernel.is_supported()[0], reason="requires native SM120")


@torch.inference_mode()
def test_fused_mlp_compiled_graph(monkeypatch):
    import mxfp6.mxfp8 as mx
    from vllm_mach.mxfp6 import mxfp8_mlp  # registers the custom op

    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "native")
    torch.manual_seed(71)
    n, k = 2560, 9216
    w = mx.quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
    values = w.dequantized_values()
    mx.begin_workspace_planning()
    for m in (1, 16, 32, 128):
        mx.warmup(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16), w)
    mx.finalize_workspace_planning()

    def fused(x):
        return torch.ops.vllm.mach_swiglu_mxfp8_down(x, values, w.scales)

    compiled = torch.compile(fused, fullgraph=True, dynamic=True)
    for m in (1, 16, 32, 128):
        x = torch.randn(m, 2*k, device="cuda", dtype=torch.bfloat16)
        # Exercise the compiled fake implementation before capturing.
        compiled(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mx.warmup(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16), w)
            compiled(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = compiled(x)
            for zero in (False, False, True):
                x.zero_() if zero else x.normal_()
                graph.replay()
                gate, up = x.chunk(2, dim=-1)
                activation = torch.nn.functional.silu(gate) * up
                reference = mx.gemm_from_float(activation.contiguous(), w)
                error = (out.float()-reference.float()).norm()
                relative = error / reference.float().norm().clamp_min(1e-10)
                assert torch.isfinite(out).all()
                assert relative.item() < 0.005
        torch.cuda.current_stream().wait_stream(stream)


def test_compiled_gdn_reads_runtime_context(monkeypatch):
    from types import SimpleNamespace
    import vllm.forward_context
    from vllm.config import CompilationMode
    from vllm_mach.mxfp6 import gdn_decode

    owner = torch.nn.Module()
    config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE, splitting_ops=[])
    owner.vllm_config = SimpleNamespace(compilation_config=config)
    assert gdn_decode._prepare_compiled_dispatch(owner)
    assert gdn_decode._prepare_compiled_dispatch(owner)
    assert config.splitting_ops == ['vllm::mach_gdn_forward']
    layer = SimpleNamespace(_mach_gdn_original=None, _mach_gdn_persistent=True,
                            _mach_gdn_aux=None)
    context = SimpleNamespace(no_compile_layers={'test': layer}, decode=False)
    monkeypatch.setattr(vllm.forward_context, 'get_forward_context', lambda: context)

    def runtime_forward(layer, original, persistent, aux, x):
        return x + (1 if context.decode else 2)

    monkeypatch.setattr(gdn_decode, '_forward', runtime_forward)
    compiled = torch.compile(lambda x: gdn_decode._compiled_dispatch('test', x),
                             fullgraph=True)
    x = torch.zeros(4, 2560, device='cuda', dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(x), x + 2)
    context.decode = True
    torch.testing.assert_close(compiled(x), x + 1)
    context.decode = False
    torch.testing.assert_close(compiled(x), x + 2)


@torch.inference_mode()
def test_fused_mlp_flashinfer_graph(monkeypatch):
    import mxfp6.mxfp8 as mx
    import flashinfer
    from vllm_mach.mxfp6 import mxfp8_mlp

    monkeypatch.setenv('VLLM_MACH_MXFP8_BACKEND', 'flashinfer')
    w = mx.quantize_mxfp8(torch.randn(2560, 9216, device='cuda', dtype=torch.bfloat16))
    for m in (1, 16):
        x = torch.randn(m, 18432, device='cuda', dtype=torch.bfloat16)
        fn = lambda: mxfp8_mlp._down(x, w.dequantized_values(), w.scales)
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        for zero in (False, True):
            x.zero_() if zero else x.normal_()
            graph.replay()
            gate, up = x.chunk(2, dim=-1)
            values, scales = torch.ops.mxfp6.quantize_mxfp8(
                (torch.nn.functional.silu(gate) * up).contiguous())
            reference = flashinfer.mm_mxfp8(
                values.view(m, 9216).view(torch.float8_e4m3fn),
                w.dequantized_values().T, scales, w.scales,
                out_dtype=torch.bfloat16, backend='cutlass')
            error = (out.float() - reference.float()).norm()
            assert (error / reference.float().norm().clamp_min(1e-10)).item() < .005
