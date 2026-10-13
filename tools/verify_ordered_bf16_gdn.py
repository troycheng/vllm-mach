#!/usr/bin/env python3
"""Compare ordered BF16/FP32 decode with stock; time complete W4 graph cycles."""
import argparse
import importlib
import json
from pathlib import Path

import torch
import triton
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode as stock,
)


@torch.inference_mode()
def run(model, m, dtype, steps, failure_dir=None, snapshot_before_step=False):
    name = ('vllm_mach.mxfp8.gdn.ordered_full_m4_m16_deferred_dispatch'
            if model == '4b' else 'vllm_mach.mxfp8.two_b.gdn.dispatch')
    op = importlib.import_module(name)
    hv = 32 if model == '4b' else 16
    width = 4096 + hv * 128
    torch.manual_seed(731)
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device='cuda', dtype=dtype) * .1
    # Real QKV row stride and padded state page stride, with untouched canaries.
    qkv = rand(m, width + hv * 128)[:, :width]
    a, b = rand(m, hv), rand(m, hv)
    al, dt = rand(hv, dtype=torch.float32) - 2, rand(hv)
    storage = rand(m + 3, hv * 128 * 128 + 128, dtype=dtype)
    base = storage[:, :hv * 128 * 128].view(m + 3, hv, 128, 128)
    ref_storage = storage.clone()
    ref = ref_storage[:, :hv * 128 * 128].view(m + 3, hv, 128, 128)
    initial = storage.clone()
    ids = torch.arange(1, m + 1, device='cuda', dtype=torch.int32)
    out = torch.empty(m, 1, hv, 128, device='cuda', dtype=torch.bfloat16)
    expected = torch.empty_like(out)
    scratch = op.DeferredScratch(**{
        k: torch.zeros(s, device='cuda', dtype=torch.int32 if k == 'age' else torch.float32)
        for k, s in op.required_shapes(m + 3, 4).items()})
    def ordered():
        op.decode(qkv, a, b, al, dt, base, out, ids, scratch)
    def reference():
        stock(qkv, a, b, al, dt, 128 ** -.5, ref, expected, ids, True)
    before = None
    def check(actual, wanted, stage):
        try:
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0,
                msg=lambda message: f'{model}, M={m}, step={step}, {stage}: {message}')
        except AssertionError:
            if failure_dir is not None:
                failure_dir.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model=model, rows=m, step=step, stage=stage,
                    before=before, qkv=qkv, a=a, b=b, al=al, dt=dt, ids=ids,
                    storage=storage, reference_storage=ref_storage,
                    scratch=vars(scratch), out=out, expected=expected),
                    failure_dir / f'{model}-m{m}-{dtype}-step{step}-{stage}.pt')
            raise
    ordered()
    storage.copy_(initial)
    scratch.age.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ordered()
    storage.copy_(initial)
    scratch.age.zero_()
    errors, state_errors = [], []
    for step in range(steps):
        qkv.copy_(rand(m, width))
        a.copy_(rand(m, hv))
        b.copy_(rand(m, hv))
        # Changing indices exercises capture, null padding, and asynchronous ages.
        ids.copy_(torch.arange(1, m + 1, device='cuda', dtype=torch.int32).roll(step % m))
        if step % 7 == 2:
            ids[0] = 0
        if snapshot_before_step:
            # Diagnostic only: extra copies change timing and memory traffic.
            before = dict(storage=storage.clone(), reference_storage=ref_storage.clone(),
                          scratch={k: v.clone() for k, v in vars(scratch).items()})
        graph.replay()
        reference()
        errors.append(float((out.float() - expected.float()).abs().max()))
        check(out, expected, 'output')
        # Materialize at all four ages, and leave longer stretches deferred.
        if step % 13 in (0, 2, 5, 9):
            all_ids = torch.arange(1, m + 1, device='cuda', dtype=torch.int32)
            op.materialize_slots(base, all_ids, scratch)
            state_errors.append(float((base.float() - ref.float()).abs().max()))
            check(base, ref, 'materialized-state')
            # Stock fallback updates the now-dense pool before ordered resumes.
            stock(qkv, a, b, al, dt, 128 ** -.5, base, out, ids, True)
            reference()
            check(out, expected, 'stock-fallback-output')
    all_ids = torch.arange(1, m + 1, device='cuda', dtype=torch.int32)
    op.materialize_slots(base, all_ids, scratch)
    check(base, ref, 'final-state')
    assert torch.equal(storage[:, -128:], initial[:, -128:])
    assert torch.equal(base[0], initial[0, :hv * 128 * 128].view(hv, 128, 128))
    assert torch.equal(base[-2:], initial[-2:, :hv * 128 * 128].view(2, hv, 128, 128))
    op.reset_slots(base, all_ids, scratch, zero_base=True)
    assert not base[1:m+1].count_nonzero()
    assert not scratch.age.count_nonzero()
    ids.copy_(all_ids)
    # Four calls per graph unit cover all ages, including the flush cost.
    def cycle(fn):
        for _ in range(4):
            fn()
    ordered_us = triton.testing.do_bench_cudagraph(lambda: cycle(ordered), rep=100) * 1000 / 4
    stock_us = triton.testing.do_bench_cudagraph(lambda: cycle(reference), rep=100) * 1000 / 4
    return dict(model=model, m=m, dtype=str(dtype), steps=steps,
                output_max_abs=max(errors), state_max_abs=max(state_errors),
                ordered_us=ordered_us, stock_us=stock_us, speedup=stock_us/ordered_us)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--models', nargs='+', choices=['4b', '2b'], default=['4b', '2b'])
    p.add_argument('--rows', nargs='+', type=int)
    p.add_argument('--dtypes', nargs='+', default=['bfloat16', 'float32'])
    p.add_argument('--failure-dir', type=Path,
                   help='save inputs, outputs and state when an equality check fails')
    p.add_argument('--snapshot-before-step', action='store_true',
                   help='also save pre-step state; diagnostic copies alter timing')
    args = p.parse_args()
    results = []
    for model in args.models:
        for dtype in args.dtypes:
            rows = args.rows or ([1, 2, 4, 8, 16, 24, 32, 64, 128] if model == '4b'
                                 else ([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160]
                                       if dtype == 'bfloat16' else [32, 48, 64, 96, 128, 160]))
            for m in rows:
                row = run(model, m, getattr(torch, dtype), args.steps,
                          args.failure_dir, args.snapshot_before_step)
                results.append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                                                       torch_version=torch.__version__, triton_version=triton.__version__,
                                                       timing='single pool, CUDA graph, complete W4 cycles, 100ms',
                                                       results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
