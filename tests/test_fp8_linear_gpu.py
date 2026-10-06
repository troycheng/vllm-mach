# SPDX-License-Identifier: Apache-2.0
"""GPU byte/graph contracts for the qualified block-FP8 GEMM boundary."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm import _custom_ops as ops
from vllm_mach.fp8 import linear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="requires the qualified CUDA SM120 target",
)


@pytest.fixture(scope="module", autouse=True)
def registered_backend():
    linear.install(n64=True, ordered=True)
    yield
    linear.uninstall()


def tensors(m, k):
    generator = torch.Generator(device="cuda").manual_seed(173 + m + k)
    a = torch.randn((m, k), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    b = torch.randn((2560, k), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    sa = torch.rand((k // 128, m), device="cuda", generator=generator).T
    sb = torch.rand((20, k // 128), device="cuda", generator=generator)
    return a, b, sa, sb, sb.repeat_interleave(2, dim=0)


def reference(a, b, sa, sb):
    return ops.cutlass_scaled_mm(
        a, b.T, out_dtype=torch.bfloat16, scale_a=sa, scale_b=sb.T
    )


def assert_bytes(actual, expected):
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


@pytest.mark.parametrize("m", [1, 4, 8, 9, 16, 32, 64, 128])
@pytest.mark.parametrize("k", [4096, 9216])
def test_selected_routes_and_m9_fallback_keep_stock_output_bytes(m, k):
    a, b, sa, sb, sb64 = tensors(m, k)
    expected = reference(a, b, sa, sb)
    actual = linear.gemm(a, b, sa, sb, sb64, n64=True, ordered=True)
    assert_bytes(actual, expected)
    assert_bytes(sb64[::2], sb)
    assert_bytes(sb64[1::2], sb)


@pytest.mark.parametrize("m", [4, 32])
def test_separate_graphs_have_independent_storage_and_changing_input_bytes(m):
    a, b, sa, sb, sb64 = tensors(m, 4096)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            linear.gemm(a, b, sa, sb, sb64, n64=True, ordered=True)
    torch.cuda.current_stream().wait_stream(warmup)
    graphs, outputs = [], []
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = linear.gemm(a, b, sa, sb, sb64, n64=True, ordered=True)
        graphs.append(graph)
        outputs.append(output)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    for shift in [1, 2, 3]:
        a.copy_(a.float().roll(shift, dims=0).to(a.dtype))
        expected = reference(a, b, sa, sb)
        for graph, output in zip(graphs, outputs):
            graph.replay()
            assert_bytes(output, expected)


@pytest.mark.parametrize("m", [4, 32])
def test_independent_stream_calls_do_not_share_scratch_or_outputs(m):
    a, b, sa, sb, sb64 = tensors(m, 4096)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(
                linear.gemm(a, b, sa, sb, sb64, n64=True, ordered=True)
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    expected = reference(a, b, sa, sb)
    for output in outputs:
        assert_bytes(output, expected)
