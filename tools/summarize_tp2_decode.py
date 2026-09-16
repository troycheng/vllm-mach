#!/usr/bin/env python3
"""Summarize per-rank P0 traces without adding rank or overlapping stream time."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def interval_union(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('Negative interval')
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def category(name):
    if name == "_producer":
        return "gdn_norm_quantization"
    if "elementwise_kernel<128, 4" in name and "direct_copy_kernel_cuda" in name:
        return "strided_tensor_copy"
    if "quantize_mx_kernel" in name and ", true>(" in name:
        return "swiglu_quantization"
    if "cutlass" in name:
        if "Sm120" in name or "sm120" in name:
            return "mxfp6_gemm"
        if "bf16" in name:
            return "bf16_gemm"
        return "other_cutlass"
    for fragment, group in [('quantize_mx_kernel', 'activation_quantization'),
                            ('memset', 'buffer_initialization'),
                            ('allreduce_fusion', 'allreduce_residual_norm'),
                            ('gdn_fused_decode', 'persistent_gdn'),
                            ('act_and_mul', 'swiglu'),
                            ('layer_norm_fwd', 'gated_norm'),
                            ('fused_recurrent_gated_delta_rule', 'gdn_recurrence'),
                            ('causal_conv1d_update', 'gdn_convolution'),
                            ('splitKreduce_kernel', 'bf16_splitk_reduction'),
                            ('gemvx', 'bf16_gemv'),
                            ('unified_attention', 'attention')]:
        if fragment in name:
            return group
    return 'other'


def summarize_trace(path):
    events = [e for e in json.loads(path.read_text())['traceEvents']
              if e.get('cat') in ('kernel', 'gpu_memset', 'gpu_memcpy') and 'dur' in e]
    groups = defaultdict(list)
    for e in events:
        groups[category(e['name'])].append(e)
    return dict(gpu_activity_union_us=interval_union([(e['ts'], e['ts']+e['dur']) for e in events]),
                categories={k:dict(count=len(v), summed_us=sum(e['dur'] for e in v),
                                  union_us=interval_union([(e['ts'],e['ts']+e['dur']) for e in v]))
                            for k,v in groups.items()})


def collect(baseline, profile):
    output = dict(contract=json.loads((baseline/'contract.json').read_text()),
                  profile_contract=json.loads((profile/'contract.json').read_text()), rows=[])
    for row in json.loads((baseline/'summary.json').read_text()):
        m = row['rows']
        metrics = [v for t in row['trials'] for v in t['metrics']]
        assert all(v and v['num_generation_tokens'] > 1 for v in metrics)
        row['mean_request_itl_ms'] = statistics.mean(
            1000*(v['last_token_ts']-v['first_token_ts'])/(v['num_generation_tokens']-1) for v in metrics)
        row['mean_request_prefill_ms'] = statistics.mean(1000*(v['first_token_ts']-v['scheduled_ts']) for v in metrics)
        trial = profile/f'm{m}-trial0'
        data = json.loads((trial/'result.json').read_text())
        assert {r['rank'] for r in data['ranks']} == {0,1}
        ranks = []
        for rank in data['ranks']:
            samples = rank['samples']
            assert len(samples)==3 and all(s['logical_rows']==s['padded_rows']==m for s in samples)
            shapes = Counter()
            for layer in rank['inventory']:
                if layer['kernel'] == 'Mxfp6Sm120LinearKernel':
                    n, pk = layer['weight_shape']
                    shapes[f'N{n}_K{pk*4//3}'] += 1
            ranks.append(dict(rank=rank['rank'], trace=summarize_trace(trial/f"rank{rank['rank']}.json"),
                measured_decode_steps=3, physical_rows=m, logical_rows=m,
                fused_ar_norm_modules=sum(v['fused_ar_norm'] is True for v in rank['inventory']),
                fused_gdn_quant_layers=rank.get('fused_gdn_quant_layers',0),
                empty_output_layers=rank.get('empty_output_layers',0),
                recurrent_tile8_layers=rank.get('recurrent_tile8_layers',0),
                shapes=dict(shapes), gdn=rank['gdn'], fused_swiglu_layers=rank.get('fused_swiglu_layers',0), peak_allocated_bytes=rank['peak_allocated_bytes'],
                peak_reserved_bytes=rank['peak_reserved_bytes'],
                mean_inclusive_module_ms={name:statistics.mean(s['inclusive_ms'][name] for s in samples)
                                          for name in samples[0]['inclusive_ms']}))
        row['profile_ranks'] = ranks
        output['rows'].append(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(collect(a.baseline,a.profile),indent=2)+'\n')


if __name__=='__main__':
    main()
