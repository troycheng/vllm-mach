"""Exact strided BA consumption across auxiliary-stream graph replays."""
from itertools import product

import pytest
import torch


@pytest.mark.parametrize('rows,dtype,layout', product(
    (16, 24, 32), (torch.float32, torch.float16), ('SD', 'DS')))
@torch.inference_mode()
def test_strided_ba_graph(rows, dtype, layout):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode as recurrent,
    )
    torch.manual_seed(7100 + rows)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device='cuda', dtype=dtype) * 0.1

    qkv = rand(rows, 8192)[:, :5120]
    hidden, weight = rand(rows, 5120), rand(48, 5120)
    conv_weight, conv_bias = rand(5120, 4), rand(5120)
    al, dt = rand(24, dtype=torch.float32), rand(24)
    indices = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32)
    conv_indices = indices.clone()
    state0 = rand(rows + 2, 24, 128, 128, dtype=dtype)
    conv0 = rand(rows + 2, 3, 5120).transpose(1, 2)
    if layout == 'DS':
        conv0 = conv0.contiguous()
    states = [state0.clone() for _ in range(3)]
    convs = [conv0.clone() for _ in range(3)]
    outputs = [torch.empty(rows, 1, 24, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    scratch = [torch.empty_like(qkv) for _ in range(3)]
    aux = torch.cuda.Stream()
    capture_stream = torch.cuda.Stream()

    def call(index, copy):
        main = torch.cuda.current_stream()
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            ba = torch.nn.functional.linear(hidden, weight)
            b, a = ba.chunk(2, -1)
            if copy:
                b, a = b.contiguous(), a.contiguous()
            assert a.stride(0) == (24 if copy else 48)
        cq = causal_conv1d_update(qkv, convs[index], conv_weight, conv_bias,
            'silu', conv_state_indices=conv_indices, validate_data=False, out=scratch[index])
        main.wait_stream(aux)
        ba.record_stream(main)
        b.record_stream(main)
        a.record_stream(main)
        recurrent(mixed_qkv=cq, a=a, b=b, A_log=al, dt_bias=dt,
            scale=128**-0.5, initial_state=states[index], out=outputs[index],
            ssm_state_indices=indices, use_qk_l2norm_in_kernel=True)

    for i in range(3):
        call(i, i == 1)
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for i, graph in enumerate(graphs):
        with torch.cuda.graph(graph, stream=capture_stream):
            call(i, i == 1)
    for s, c in zip(states, convs):
        s.copy_(state0)
        c.copy_(conv0)
    for step in range(120):
        qkv.copy_(rand(rows, 5120))
        hidden.copy_(rand(rows, 5120))
        # Rotate live slots and then recycle the omitted slot after padding.
        indices.copy_(torch.arange(1, rows + 1, device='cuda', dtype=torch.int32).roll(step % rows))
        if step % 3:
            indices[0] = 0 if step % 3 == 1 else -1
        # vLLM convolution uses null slot 0; packed recurrence also accepts -1.
        conv_indices.copy_(indices.clamp_min(0))
        for graph in graphs:
            graph.replay()
        call(2, False)
        torch.cuda.synchronize()
        for i in (1, 2):
            assert torch.equal(outputs[0], outputs[i]), (step, 'output')
            assert torch.equal(states[0], states[i]), (step, 'state')
            assert torch.equal(convs[0], convs[i]), (step, 'conv')
        for s, c in zip(states, convs):
            assert torch.equal(s[0], state0[0]) and torch.equal(s[-1], state0[-1])
            assert torch.equal(c[0], conv0[0]) and torch.equal(c[-1], conv0[-1])
        if step % 3:
            assert torch.count_nonzero(outputs[0][0]) == 0
        assert torch.isfinite(states[0]).all()
