# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check packed checkpoint loading and changing-input CUDA Graph execution."""

import pytest
import torch
from vllm.model_executor.kernels.linear.mxfp6.base import MxFp6LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)

from vllm_mach.mxfp6.dense import (
    Mxfp6Sm120LinearKernel,
    is_mxfp6_sm120_available,
)
from vllm_mach.mxfp6.warmup import (
    warmup_mxfp6_sm120,
    warmup_mxfp6_sm120_stream,
)

pytestmark = pytest.mark.skipif(
    not is_mxfp6_sm120_available(), reason="requires mxfp6-sm120 and SM120"
)


@torch.inference_mode()
def test_checkpoint_scales_and_changing_graph_input():
    """The loaded scale layout and captured workspace must survive new inputs."""
    from types import SimpleNamespace

    import mxfp6

    dtype = torch.bfloat16
    torch.manual_seed(17)
    n, k = 512, 512
    packed = mxfp6.quantize_mxfp6(torch.randn(n, k, device="cuda", dtype=dtype))
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed.values.reshape(n, k * 3 // 4), False)
    layer.weight_scale = torch.nn.Parameter(
        mxfp6.unpack_scales(packed.scales, n, k), False
    )
    kernel = Mxfp6Sm120LinearKernel(
        MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    )
    layer.scheme = SimpleNamespace(ocp_mx_linear=kernel)
    kernel.process_weights_after_loading(layer)
    warmup_mxfp6_sm120(layer, [32], dtype)
    x = torch.randn(2, 16, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup_mxfp6_sm120_stream(layer, [32], dtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = kernel.apply_weights(layer, x, bias)
        for _ in range(3):
            x.normal_()
            graph.replay()
            expected = (
                mxfp6.gemm_from_float(
                    x.reshape(32, k), packed, out_dtype=dtype
                ).reshape(2, 16, n)
                + bias
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
