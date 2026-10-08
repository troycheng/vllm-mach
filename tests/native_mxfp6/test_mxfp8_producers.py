"""Check native PDL dependencies and fused producer rounding under graph replay."""
import pytest
import torch

from vllm_mach.mxfp6.dense_mxfp8 import Mxfp8Sm120LinearKernel

pytestmark = pytest.mark.skipif(
    not Mxfp8Sm120LinearKernel.is_supported()[0], reason="native SM120 required")


@pytest.mark.parametrize("pdl", [False, True])
@torch.inference_mode()
def test_norm_quant_changed_graph_input(monkeypatch, pdl):
    import mxfp6.mxfp8 as mx

    from vllm_mach.mxfp6 import gemma_norm
    from vllm_mach.mxfp6 import mxfp8_norm_quant as producer

    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "native")
    monkeypatch.setenv("VLLM_MACH_MXFP8_PDL", str(int(pdl)))
    for m in (0, 1, 16, 32, 513):
        x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
        producer.quantize(x, residual, weight, 1e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            a, scales, summed = producer.quantize(x, residual, weight, 1e-6)
        for zero in (False, False, True):
            x.zero_() if zero else x.normal_()
            residual.zero_() if zero else residual.normal_()
            weight.normal_()
            graph.replay()
            norm, ref_sum = gemma_norm._impl(x, residual, weight, 1e-6)
            assert torch.equal(summed, ref_sum)
            if m:
                ref = mx.quantize_mxfp8(norm)
                actual_scale = mx.unpack_scales(scales, m, 2560).float()
                ref_scale = mx.unpack_scales(ref.scales, m, 2560).float()
                actual = a.float() * torch.exp2(actual_scale - 127).repeat_interleave(32, 1)
                expected = ref.dequantized_values().float() * torch.exp2(ref_scale - 127).repeat_interleave(32, 1)
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-10)
                # Fusing the reductions can change BF16 values at rounding ties.
                assert relative.item() < .002


@torch.inference_mode()
def test_pdl_gemm_changing_inputs(monkeypatch):
    import mxfp6.mxfp8 as mx

    from vllm_mach.mxfp6 import (
        dense_mxfp8,
        mxfp8_mlp,
        mxfp8_norm_quant,  # noqa: F401 - registers the op
    )

    if not hasattr(torch.ops.mxfp8_sm120, "pdl_version"):
        pytest.skip("use both rebuilt PDL version 3 libraries")
    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "native")
    monkeypatch.setenv("VLLM_MACH_MXFP8_PDL", "1")
    weights = {shape: mx.quantize_mxfp8(torch.randn(*shape, device="cuda", dtype=torch.bfloat16))
               for shape in sorted(dense_mxfp8._B12X_SHAPES)}
    planning = not mx.workspace_stats().get("frozen", 0)
    if planning:
        mx.begin_workspace_planning()
    for (n, k), w in weights.items():
        for m in (1, 16, 32, 128):
            mx.warmup(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16), w)
    if planning:
        mx.finalize_workspace_planning()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for (n, k), w in weights.items():
            for m in (1, 16, 32, 128):
                mx.warmup(torch.zeros(m, k, device="cuda", dtype=torch.bfloat16), w)
    torch.cuda.current_stream().wait_stream(stream)
    for (n, k), w in weights.items():
        for m in (1, 16, 32, 128):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            dense_mxfp8._gemm(x, w.dequantized_values(), w.scales)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y = dense_mxfp8._gemm(x, w.dequantized_values(), w.scales)
            for zero in (False, False, True):
                x.zero_() if zero else x.normal_()
                graph.replay()
                ref = torch.ops.mxfp8_sm120.gemm_from_float(x, w.dequantized_values(), w.scales)
                assert torch.equal(y, ref)
    w = weights[(18432, 2560)]
    down = weights[(2560, 9216)]
    norm_weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(lambda x, r: torch.ops.vllm.mach_norm_quant_mxfp8_gate_up(
        x, r, norm_weight, 1e-6, w.dequantized_values(), w.scales), fullgraph=True, dynamic=True)
    for m in (1, 16, 32, 128):
        x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        compiled(x, r)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            up, summed = compiled(x, r)
            y = mxfp8_mlp._down(up, down.dequantized_values(), down.scales)
        for zero in (False, False, True):
            x.zero_() if zero else x.normal_()
            r.zero_() if zero else r.normal_()
            graph.replay()
            from vllm_mach.mxfp6.gemma_norm import _impl
            norm, ref_sum = _impl(x, r, norm_weight, 1e-6)
            ref_up = mx.gemm_from_float(norm, w)
            ref_y = mx.gemm_from_float(torch.nn.functional.silu(ref_up.chunk(2, -1)[0])
                                      * ref_up.chunk(2, -1)[1], down)
            relative_up = (up.float() - ref_up.float()).norm() / ref_up.float().norm().clamp_min(1e-10)
            assert relative_up.item() < .005
            assert torch.equal(summed, ref_sum)
            relative = (y.float() - ref_y.float()).norm() / ref_y.float().norm().clamp_min(1e-10)
            assert relative.item() < .005


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_native_pdl_quantizer_replay(dtype):
    import mxfp6.mxfp8 as mx

    mx.load_library()
    if not hasattr(torch.ops.mxfp6, "quantize_mxfp8_pdl"):
        pytest.skip("use both rebuilt native libraries")
    for m in (1, 16, 32, 128):
        for k in (2560, 4096, 9216):
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            torch.ops.mxfp6.quantize_mxfp8_pdl(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                a, scales = torch.ops.mxfp6.quantize_mxfp8_pdl(x)
            for zero in (False, False, True):
                x.zero_() if zero else x.normal_()
                a.fill_(255)
                scales.fill_(255)
                graph.replay()
                ref_a, ref_scales = torch.ops.mxfp6.quantize_mxfp8(x)
                assert torch.equal(a, ref_a)
                assert torch.equal(scales, ref_scales)


@torch.inference_mode()
def test_gemm_norm_quant_swiglu_dependency_chain(monkeypatch):
    """Exercise GEMM -> norm -> quant/GEMM -> SwiGLU/quant -> GEMM twice.

    In contrast to quantizing external input, every consumer here reads a
    buffer written by a preceding PDL kernel in the same captured graph.
    """
    import mxfp6.mxfp8 as mx

    from vllm_mach.mxfp6 import dense_mxfp8, gemma_norm, mxfp8_mlp

    mx.load_library()
    if not hasattr(torch.ops.mxfp6, "quantize_mxfp8_pdl"):
        pytest.skip("use both rebuilt native libraries")
    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "native")
    up = mx.quantize_mxfp8(torch.randn(18432, 2560, device="cuda", dtype=torch.bfloat16))
    down = mx.quantize_mxfp8(torch.randn(2560, 9216, device="cuda", dtype=torch.bfloat16))
    weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    planning = not mx.workspace_stats().get("frozen", 0)
    if planning:
        mx.begin_workspace_planning()
    for m in (1, 16, 32):
        for w in (up, down):
            mx.warmup(torch.zeros(m, w.shape[1], device="cuda", dtype=torch.bfloat16), w)
    if planning:
        mx.finalize_workspace_planning()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for m in (1, 16, 32):
            for w in (up, down):
                mx.warmup(torch.zeros(m, w.shape[1], device="cuda", dtype=torch.bfloat16), w)
    torch.cuda.current_stream().wait_stream(stream)

    def chain(x, r):
        y = dense_mxfp8._gemm(x, down.dequantized_values(), down.scales)
        for _ in range(2):
            norm, r = gemma_norm._impl(y, r, weight, 1e-6)
            gate_up = dense_mxfp8._gemm(norm, up.dequantized_values(), up.scales)
            y = mxfp8_mlp._down(gate_up, down.dequantized_values(), down.scales)
        return y, r

    for m in (1, 16, 32):
        x = torch.randn(m, 9216, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
        graphs, outputs = [], []
        for pdl in (False, True):
            monkeypatch.setenv("VLLM_MACH_MXFP8_PDL", str(int(pdl)))
            with torch.cuda.stream(stream):
                chain(x, r)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs.append(chain(x, r))
            graphs.append(graph)
        for zero in (False, False, False, True):
            x.zero_() if zero else x.normal_()
            r.zero_() if zero else r.normal_()
            weight.normal_()
            for graph in graphs:
                graph.replay()
            for baseline, dependent in zip(*outputs):
                assert torch.equal(baseline, dependent)
