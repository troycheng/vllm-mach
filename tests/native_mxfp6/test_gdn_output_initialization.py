"""Prove GDN producers overwrite every output element before consumption."""
from itertools import product

import pytest
import torch


@pytest.mark.parametrize('layout', ('SD', 'DS'))
@pytest.mark.parametrize('rows,dtype', list(product(
    (1, 2, 4, 8, 16, 24, 32), (torch.float32, torch.float16))))
@torch.inference_mode()
def test_output_write_before_read(rows, dtype, layout):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from vllm_mach.mxfp6.gdn import persistent
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode as recurrent,
    )
    torch.manual_seed(7400 + rows)
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device='cuda', dtype=dtype) * .1
    hidden = rand(rows, 5120)
    ba_weight = rand(48, 5120).T.contiguous()
    qkv = rand(rows, 8192)[:, :5120]
    cw, cb = rand(5120, 4), rand(5120)
    al, dt = rand(24, dtype=torch.float32), rand(24)
    ba = rand(rows, 48)
    b, a = [x.contiguous() for x in ba.chunk(2, -1)]
    idx = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32)
    state0 = rand(rows + 2, 24, 128, 128, dtype=dtype)
    conv0 = rand(rows + 2, 3, 5120).transpose(1, 2)
    if layout == 'DS':
        conv0 = conv0.contiguous()
    states, convs = [state0.clone() for _ in range(2)], [conv0.clone() for _ in range(2)]
    storage = [torch.full((rows + 2, 1, 24, 128), 19., device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    outputs = [o[1:-1] for o in storage]
    def call(i):
        if rows <= 8:
            persistent.execute(hidden, ba_weight, qkv, cw, cb, convs[i], al, dt,
                               128**-.5, states[i], idx, outputs[i])
        else:
            recurrent(mixed_qkv=qkv, a=a, b=b, A_log=al, dt_bias=dt,
                      scale=128**-.5, initial_state=states[i], out=outputs[i],
                      ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    for i in range(2):
        call(i)
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for i, graph in enumerate(graphs):
        with torch.cuda.graph(graph):
            call(i)
    for state, conv in zip(states, convs):
        state.copy_(state0)
        conv.copy_(conv0)
    for step in range(120):
        qkv.copy_(rand(rows, 5120))
        hidden.copy_(rand(rows, 5120))
        a.copy_(rand(rows, 24))
        b.copy_(rand(rows, 24))
        idx.copy_(torch.arange(1, rows + 1, device='cuda', dtype=torch.int32).roll(step % rows))
        if step % 4:
            idx[0] = 0 if step % 4 == 1 else -1
        if step % 4 == 3:
            idx.fill_(0)  # All-padding launch must also fully overwrite output.
        outputs[0].zero_()
        outputs[1].fill_(float('nan'))
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[0], outputs[1]), (rows, step, 'output')
        assert torch.isfinite(outputs[1]).all()
        assert torch.equal(states[0], states[1])
        assert torch.equal(convs[0], convs[1])
        for backing, state, conv in zip(storage, states, convs):
            assert (backing[0] == 19).all() and (backing[-1] == 19).all()
            assert torch.equal(state[0], state0[0]) and torch.equal(state[-1], state0[-1])
            assert torch.equal(conv[0], conv0[0]) and torch.equal(conv[-1], conv0[-1])
        if step % 4:
            assert torch.count_nonzero(outputs[1][0]) == 0
