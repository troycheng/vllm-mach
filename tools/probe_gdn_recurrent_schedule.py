#!/usr/bin/env python3
"""Bounded packed GDN schedule screening; never a full-model speedup estimate.

This deliberately reuses one state allocation. Use full-checkpoint decode and
serving measurements for acceptance, because its cache residency is different.
"""
import argparse
import hashlib
import inspect
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as reference,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel as kernel,
)


def launch(q, a, b, al, dt, state, out, indices, tile, warps):
    kernel[(128 // tile, q.shape[0] * 24)](
        mixed_qkv=q, a=a, b=b, A_log=al, dt_bias=dt, o=out,
        h0=state, ht=state, ssm_state_indices=indices, scale=128**-.5,
        stride_mixed_qkv_tok=q.stride(0), stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0), stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0), stride_indices_seq=1,
        H=8, HV=24, K=128, V=128, BK=128, BV=tile,
        SOFTPLUS_THRESHOLD=20., USE_QK_L2NORM_IN_KERNEL=True,
        SPLIT_BATCH_HEAD_GRID=False, num_warps=warps, num_stages=3)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(718)
    results = []
    for rows in (16, 24, 32):
        q = torch.randn(rows, 8192, device='cuda', dtype=torch.bfloat16)[:, :5120]
        a = torch.randn(rows, 24, device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        al = torch.randn(24, device='cuda')
        dt = torch.randn_like(a[0])
        indices = torch.arange(1, rows + 1, device='cuda', dtype=torch.int32)
        state = torch.randn(rows + 2, 24, 128, 128, device='cuda') * .1
        expected_state = state.clone()
        output = torch.empty(rows, 1, 24, 128, device='cuda', dtype=torch.bfloat16)
        expected_output = torch.empty_like(output)
        reference(mixed_qkv=q, a=a, b=b, A_log=al, dt_bias=dt,
                  scale=128**-.5, initial_state=expected_state, out=expected_output,
                  ssm_state_indices=indices, use_qk_l2norm_in_kernel=True)
        for tile, warps in ((8, 1), (16, 1), (32, 1), (64, 1),
                            (16, 2), (32, 2), (32, 4), (64, 4)):
            candidate = state.clone()
            call = partial(launch, q, a, b, al, dt, candidate, output, indices, tile, warps)
            call()
            record = {'rows': rows, 'tile': tile, 'warps': warps,
                'output_equal': torch.equal(output, expected_output),
                'state_equal': torch.equal(candidate, expected_state),
                'state_max_abs': (candidate - expected_state).abs().max().item(),
                'output_mismatches': int(torch.count_nonzero(output != expected_output)),
                'state_mismatches': int(torch.count_nonzero(candidate != expected_state))}
            samples = []
            for _ in range(5):
                candidate.copy_(state)
                samples.append(1000 * do_bench_cudagraph(call, rep=100))
            record.update(samples_us=samples, median_us=statistics.median(samples))
            results.append(record)
            print(record, flush=True)
    source = Path(inspect.getfile(reference))
    data = {'schema': 'mach-gdn-recurrent-schedule-probe/v1',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'Single-layer cached-state screening; numerical checks are single-step. '
              'Changing-input replay and real-model validation are separate.',
        'results': results}
    args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
