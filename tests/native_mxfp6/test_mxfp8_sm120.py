# SPDX-License-Identifier: Apache-2.0
"""Checkpoint layout, reference math, and changing-input W8A8 graph replay."""
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig

from vllm_mach.mxfp6.dense_mxfp8 import Mxfp8Sm120LinearKernel, warmup_mxfp8

pytestmark = pytest.mark.skipif(
    not Mxfp8Sm120LinearKernel.is_supported()[0], reason="requires native W8A8 on SM120")


@torch.inference_mode()
def test_checkpoint_and_graph():
    import mxfp6.mxfp8 as runtime

    torch.manual_seed(29)
    n, k = 256, 512
    packed = runtime.quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed.dequantized_values(), False)
    scales = runtime.unpack_scales(packed.scales, n, k)
    layer.weight_scale = torch.nn.Parameter(scales, False)
    kernel = Mxfp8Sm120LinearKernel(Mxfp8LinearLayerConfig())
    layer.scheme = SimpleNamespace(kernel=kernel)
    kernel.process_weights_after_loading(layer)
    assert torch.equal(layer.weight.view(torch.uint8), packed.dequantized_values().view(torch.uint8))
    warmup_mxfp8(layer, [1, 16, 32])
    x = torch.randn(2, 8, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup_mxfp8(layer, [1, 16, 32], stream=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = kernel.apply_weights(layer, x, bias)
        for zero in (False, False, True):
            x.zero_() if zero else x.normal_()
            graph.replay()
            qa = runtime.quantize_mxfp8(x.reshape(16, k))
            sa = runtime.unpack_scales(qa.scales, 16, k)
            a = qa.dequantized_values().float() * torch.exp2(sa.float() - 127).repeat_interleave(32, dim=1)
            b = packed.dequantized_values().float() * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
            ref = (a @ b.T).bfloat16().reshape(2, 8, n) + bias
            torch.testing.assert_close(y, ref, rtol=0.01, atol=0.02)
    torch.cuda.current_stream().wait_stream(stream)
    assert runtime.workspace_stats()["fallback_launches"] == 0


def test_intermediate_prefill_workspace(monkeypatch):
    import subprocess
    import sys

    # The native arena cannot be resized once graphs have borrowed its pointers.
    # Test a separate model's initialization in a fresh worker, as serving does.
    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "native")
    subprocess.run([
        sys.executable, "-c",
        "import runpy,sys; "
        "runpy.run_path(sys.argv[1])['_check_intermediate_prefill_workspace']()",
        __file__,
    ], check=True)


@torch.inference_mode()
def _check_intermediate_prefill_workspace():
    import mxfp6.mxfp8 as runtime

    model = torch.nn.ModuleList()
    for n, k in ((2560, 4096), (2560, 9216)):
        packed = runtime.quantize_mxfp8(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(packed.dequantized_values(), False)
        layer.weight_scale = torch.nn.Parameter(packed.scales, False)
        layer.scheme = SimpleNamespace(
            kernel=Mxfp8Sm120LinearKernel(Mxfp8LinearLayerConfig()))
        model.append(layer)
    warmup_mxfp8(model, [1, 16, 32, 4096])
    before = runtime.workspace_stats()["fallback_launches"]
    for m in (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 3000, 4096):
        for layer in model:
            x = torch.zeros(m, layer.weight.shape[1], device="cuda",
                            dtype=torch.bfloat16)
            layer.scheme.kernel.apply_weights(layer, x)
    torch.cuda.synchronize()
    assert runtime.workspace_stats()["fallback_launches"] == before


@torch.inference_mode()
def test_flashinfer_decode_graph(monkeypatch):
    import mxfp6.mxfp8 as runtime
    from vllm_mach.mxfp6 import dense_mxfp8 as adapter

    monkeypatch.setenv("VLLM_MACH_MXFP8_BACKEND", "flashinfer")
    model = torch.nn.ModuleList()
    for n, k in sorted(adapter._B12X_SHAPES):
        packed = runtime.quantize_mxfp8(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(packed.dequantized_values(), False)
        layer.weight_scale = torch.nn.Parameter(packed.scales, False)
        layer.scheme = SimpleNamespace(
            kernel=Mxfp8Sm120LinearKernel(Mxfp8LinearLayerConfig()))
        model.append(layer)
    warmup_mxfp8(model, [1, 16, 32])
    for layer in model:
        for m in (1, 16, 32):
            x = torch.randn(m, layer.weight.shape[1], device="cuda",
                            dtype=torch.bfloat16)
            layer.scheme.kernel.apply_weights(layer, x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = layer.scheme.kernel.apply_weights(layer, x)
            for zero in (False, True):
                x.zero_() if zero else x.normal_()
                graph.replay()
                reference = torch.ops.mxfp8_sm120.gemm_from_float(
                    x, layer.weight, layer.weight_scale)
                error = (result.float() - reference.float()).norm()
                scale = reference.float().norm().clamp_min(1e-10)
                assert torch.isfinite(result).all()
                assert (error / scale).item() < 0.005
