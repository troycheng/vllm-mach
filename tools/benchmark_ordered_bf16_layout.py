#!/usr/bin/env python3
"""Compare stock, pre-layout BF16 ordered decode, and the current dispatch.

Rotate caller-owned pools between token steps to expose the working-set cost
hidden by a single cache-resident pool. Every graph unit covers all W4 ages.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import statistics
from pathlib import Path

import torch
import triton
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode as stock,
)


@torch.inference_mode()
def measure(model, rows, pools, repeats, rep_ms):
    from vllm_mach.mxfp8.gdn import bf16_small_gluon as gluon
    heads = 16 if model == '2b' else 32
    high = importlib.import_module(
        'vllm_mach.mxfp8.two_b.gdn.ordered_allm_triton' if model == '2b'
        else 'vllm_mach.mxfp8.gdn.ordered_allm_triton')
    op = importlib.import_module(
        'vllm_mach.mxfp8.two_b.gdn.dispatch' if model == '2b'
        else 'vllm_mach.mxfp8.gdn.ordered_full_m4_m16_deferred_dispatch')
    pools = pools or (6 if model == '2b' else 8)
    torch.manual_seed(731)
    width = 4096 + heads * 128

    def rand(*shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .1

    qkv = rand(rows, width + heads * 128)[:, :width]
    a, b = rand(rows, heads), rand(rows, heads)
    al, dt = rand(heads).float() - 2, rand(heads)
    ids = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32)
    states = []
    for _ in range(pools):
        storage = rand(rows + 3, heads * 16384 + 128)
        base = storage[:, :heads * 16384].view(rows + 3, heads, 128, 128)
        out = torch.empty(rows, 1, heads, 128, device='cuda', dtype=torch.bfloat16)
        scratch = high.DeferredScratch(**{
            k: torch.zeros(s, device='cuda',
                           dtype=torch.int32 if k == 'age' else torch.float32)
            for k, s in high.required_shapes(rows + 3, 4).items()})
        states.append((base, out, scratch))

    def reference(state):
        base, out, _ = state
        return stock(qkv, a, b, al, dt, 128**-.5, base, out, ids, True)

    def current(state):
        base, out, scratch = state
        return op.decode(qkv, a, b, al, dt, base, out, ids, scratch)

    def previous(state):
        base, out, scratch = state
        # Reproduce the immediately preceding BF16 route, including its
        # original one-row layout and <=1024 four-warp threshold.
        if rows * heads <= 1536:
            full = rows * heads <= 256
            warps = 8 if full else (4 if rows * heads <= 1024 else 2)
            return gluon._ordered_decode[(4, heads, rows)](
                qkv, a, b, al, dt, out, base, ids,
                scratch.pending_k, scratch.pending_d, scratch.coeff,
                scratch.prefix, scratch.age,
                qkv.stride(0), a.stride(0), b.stride(0), base.stride(0),
                ids.stride(0), base.shape[0], 4, 128**-.5, 16, heads,
                128, 128, 32, 4, full, 1, num_warps=warps, num_stages=3)
        return high._ordered_decode[(4, heads, rows)](
            qkv, a, b, al, dt, out, base, ids,
            scratch.pending_k, scratch.pending_d, scratch.coeff,
            scratch.prefix, scratch.age,
            qkv.stride(0), a.stride(0), b.stride(0), base.stride(0),
            ids.stride(0), base.shape[0], 4, 128**-.5, 16, heads,
            128, 128, 32, 4, num_warps=1, num_stages=3)

    # Record resources for the actual new Gluon signature, outside capture.
    base, out, scratch = states[0]
    compiled = gluon.decode(qkv, a, b, al, dt, base, out, ids, scratch)
    functions = dict(stock=reference, previous=previous, optimized=current)
    times = {key: [] for key in functions}
    order = list(functions)
    rng = random.Random(731)
    for _ in range(repeats):
        rng.shuffle(order)
        for key in order:
            for _, _, scratch in states:
                scratch.age.zero_()

            def cycle():
                for _ in range(4):
                    for state in states:
                        functions[key](state)

            times[key].append(
                triton.testing.do_bench_cudagraph(cycle, rep=rep_ms)
                * 1000 / (4 * pools))
    medians = {key: statistics.median(values) for key, values in times.items()}
    return dict(model=model, rows=rows, pools=pools, median_us=medians,
                samples_us=times, n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                speedup_vs_stock=medians['stock'] / medians['optimized'],
                speedup_vs_previous=medians['previous'] / medians['optimized'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['2b', '4b'], default=['2b', '4b'])
    parser.add_argument('--rows', nargs='+', type=int)
    parser.add_argument('--pools', type=int, default=1,
                        help='state pools to rotate; 0 uses six for 2B/eight for 4B')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--rep-ms', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1 or args.rep_ms < 1 or args.pools < 0:
        parser.error('repeats and rep-ms must be positive; pools must be nonnegative')
    valid_rows = {
        '2b': [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160],
        '4b': [1, 2, 4, 8, 16] + list(range(24, 129, 8)),
    }
    for model in args.models:
        if args.rows and any(m not in valid_rows[model] for m in args.rows):
            parser.error(f'rows must be configured pure-decode shapes for {model}')
    # This tool compares the experimental candidate with the preceding route.
    os.environ['VLLM_MACH_MXFP8_BF16_GDN_GROUPED'] = '1'
    from vllm_mach.mxfp8.gdn import bf16_small_gluon
    report = dict(
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__, triton_version=triton.__version__,
        repeats=args.repeats, rep_ms=args.rep_ms,
        stock_source_sha256=hashlib.sha256(Path(inspect.getfile(stock)).read_bytes()).hexdigest(),
        optimized_source_sha256=hashlib.sha256(Path(bf16_small_gluon.__file__).read_bytes()).hexdigest(),
        use_pdl=bf16_small_gluon._USE_PDL,
        grouped_layout=bf16_small_gluon._USE_GROUPED,
        previous='pre-layout BF16: STATE_ROWS=1; full/8 warps through 256 head-rows; '
                 'deferred/4 warps through 1024, 2 warps through 1536; '
                 'original Triton above 1536; PDL disabled',
        method='same state addresses/strides across variants; shuffled order; '
               'CUDA graph with four complete token steps across all pools; '
               'microseconds per pool per token step', results=[])
    for model in args.models:
        for rows in args.rows or valid_rows[model]:
            row = measure(model, rows, args.pools, args.repeats, args.rep_ms)
            report['results'].append(row)
            print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
