"""Qualified AOT fusion, rounding boundaries and changing-input graph replay."""
import ast
import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from vllm_mach.mxfp8 import swiglu_quant as fusion


def source_fixture(*, extra_consumer=False, quantized=True, altered_math=False):
    policy = Path(fusion.__file__).parent / 'two_b/data/full_pdl.json'
    spec = next(s for s in json.loads(policy.read_text())['kernels'] if 'silu_slice' in s['name'])
    source = spec['original_source']
    if altered_math:
        source = source.replace('tmp9 = tmp7 * tmp8', 'tmp9 = tmp7 + tmp8')
    assert not altered_math or source != spec['original_source']
    consumer = 'out = torch.ops.vllm.mxfp8_quantize.default(act, True)' if quantized else 'out = act'
    return (f"{spec['name']} = async_compile.triton({spec['name']!r}, {source!r})\n"
            'class Runner:\n'
            '    def call(self, args):\n'
            '        gate, act, count, stream = args\n'
            f"        {spec['name']}.run(gate, act, count, stream=stream)\n"
            f'        {consumer}\n'
            + ('        extra = act.clone()\n' if extra_consumer else '')
            + '        return out\n')


def test_transform_retains_kernel_literals():
    source = source_fixture()
    transformed, count = fusion.transform(source)
    assert count == 1
    assert fusion._producer_hashes(ast.parse(source)) == fusion._producer_hashes(ast.parse(transformed))
    assert '_mach_silu_quant(gate)' in transformed
    assert 'if _mach_silu_pair_0 is None:' in transformed
    assert 'torch.ops.vllm.mxfp8_quantize.default(act, True)' in transformed
    compile(transformed, '<fusion>', 'exec')


def test_bf16_consumer_is_unchanged():
    source = source_fixture(quantized=False)
    assert fusion.transform(source) == (source, 0)


def test_extra_consumer_rejected():
    with pytest.raises(RuntimeError, match='extra consumers'):
        fusion.transform(source_fixture(extra_consumer=True))


def test_changed_arithmetic_rejected():
    with pytest.raises(RuntimeError, match='Unqualified'):
        fusion.transform(source_fixture(altered_math=True))


@triton.jit
def compiled_silu_reference(X, Y, K: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    col, row = i % K, i // K
    gate = tl.load(X + row * 2 * K + col, i < N, 0).to(tl.float32)
    up = tl.load(X + row * 2 * K + K + col, i < N, 0).to(tl.float32)
    tl.store(Y + i, gate / (1.0 + libdevice.exp(-gate)) * up, i < N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('k', [6144, 9216])
@pytest.mark.parametrize('m', [1, 4, 16, 24, 32, 64])
def test_quantization_changing_graph_inputs(m, k):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('SM120 E8M0 instruction required')
    from flashinfer import mxfp8_quantize

    x = torch.randn(m, 2*k, device='cuda', dtype=torch.bfloat16)
    reference = torch.empty(m, k, device='cuda', dtype=torch.bfloat16)
    fusion.quantize(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, scales = fusion.quantize(x)
    rows = torch.arange(m, device='cuda')[:, None]
    groups = torch.arange(k//32, device='cuda')[None, :]
    offsets = (((rows//128*((k//32+3)//4)+groups//4)*32+rows%32)*4+rows%128//32)*4+groups%4
    for magnitude in [1., 0., 1.e-15, 50.]:
        x.normal_().mul_(magnitude)
        graph.replay()
        compiled_silu_reference[(triton.cdiv(m*k, 256),)](x, reference, k, m*k, 256)
        expected_values, expected_scales = mxfp8_quantize(reference, is_sf_swizzled_layout=True)
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
        assert torch.equal(scales[offsets], expected_scales.flatten()[offsets])
