#!/usr/bin/env python3
"""GPU acceptance for vendored GDN: null slots, canaries and changing graphs."""

import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

from vllm_mach.mxfp6.gdn import persistent


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=("27b", "35b"), default="27b")
    a = p.parse_args()
    hidden, heads = (5120, 24) if a.model == "27b" else (2048, 16)
    qkv_dim = (16 + heads) * 128
    torch.manual_seed(20260916)
    results = []

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype) * 0.1

    from itertools import product

    for state_dtype, layout, batch in product(
        (torch.float32, torch.float16), ("SD", "DS"), (1, 2, 4, 8)
    ):
        x, w = rand(batch, hidden), rand(2 * heads, hidden)
        wt = w.T.contiguous()
        qkv = rand(batch, qkv_dim + heads * 128)[:, :qkv_dim]
        cw, cb = rand(qkv_dim, 4), rand(qkv_dim)
        al, dt = torch.zeros(heads, device="cuda"), rand(heads)
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        conv = rand(batch + 2, 3, qkv_dim).transpose(1, 2)
        if layout == "DS":
            conv = conv.contiguous()
        state = rand(batch + 2, heads, 128, 128, dtype=state_dtype)
        original_conv, original_state = conv.clone(), state.clone()
        eager_conv, eager_state = conv.clone(), state.clone()
        ref_conv, ref_state = conv.clone(), state.clone()
        out = torch.empty(batch, 1, heads, 128, device="cuda", dtype=torch.bfloat16)
        eager_out, ref_out = torch.empty_like(out), torch.empty_like(out)
        scratch = torch.empty_like(qkv)

        def call(c, s, o):
            persistent.execute(x, wt, qkv, cw, cb, c, al, dt, 128**-0.5, s, indices, o)

        call(conv, state, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(conv, state, out)
        conv.copy_(original_conv)
        state.copy_(original_state)
        errors, state_errors = [], []
        for step in range(4):
            before_conv, before_state = conv[1].clone(), state[1].clone()
            x.copy_(rand(batch, hidden))
            qkv.copy_(rand(batch, qkv_dim))
            indices[0] = 0 if step == 1 else -1 if step == 2 else 1
            graph.replay()
            call(eager_conv, eager_state, eager_out)
            ba = torch.nn.functional.linear(x, w)
            b, aa = ba.chunk(2, -1)
            cq = causal_conv1d_update(
                qkv,
                ref_conv,
                cw,
                cb,
                "silu",
                conv_state_indices=indices.clamp_min(0),
                validate_data=False,
                out=scratch,
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=cq,
                a=aa.contiguous(),
                b=b.contiguous(),
                A_log=al,
                dt_bias=dt,
                scale=128**-0.5,
                initial_state=ref_state,
                out=ref_out,
                ssm_state_indices=indices.clamp_min(0),
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            assert torch.equal(out, eager_out)
            assert torch.equal(conv, eager_conv), (
                layout,
                batch,
                step,
                "conv",
                (conv - eager_conv).abs().max().item(),
            )
            assert torch.equal(state, eager_state), (
                layout,
                batch,
                step,
                "state",
                (state - eager_state).abs().max().item(),
            )
            assert torch.equal(conv, ref_conv)
            assert torch.equal(conv[0], original_conv[0])
            assert torch.equal(state[0], original_state[0])
            assert torch.equal(conv[-1], original_conv[-1])
            assert torch.equal(state[-1], original_state[-1])
            assert torch.isfinite(out).all() and torch.isfinite(state).all()
            rel = float(
                (out.float() - ref_out.float()).norm()
                / ref_out.float().norm().clamp_min(1e-9)
            )
            assert rel < 0.02, rel
            if step in (1, 2):
                assert torch.count_nonzero(out[0]) == 0
                assert torch.equal(conv[1], before_conv)
                assert torch.equal(state[1], before_state)
            state_rel = float(
                (state.float() - ref_state.float()).norm()
                / ref_state.float().norm().clamp_min(1e-9)
            )
            assert state_rel < 0.02, state_rel
            state_errors.append(state_rel)
            errors.append(rel)
        results.append(
            dict(
                model=a.model,
                batch=batch,
                state_dtype=str(state_dtype),
                layout=layout,
                relative_l2=errors,
                state_relative_l2=state_errors,
                graph_matches_eager=True,
                null_and_canary_unchanged=True,
            )
        )
        print(results[-1], flush=True)
    a.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
