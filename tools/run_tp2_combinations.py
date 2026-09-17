#!/usr/bin/env python3
"""Balanced TP2 empty-output/attention-gate experiments, per precision profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VARIANTS = {'control': ('0', '0'), 'empty': ('1', '0'),
            'gate': ('0', '1'), 'both': ('1', '1')}
ORDERS = [('control', 'empty', 'gate', 'both'), ('both', 'gate', 'empty', 'control')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision-profile', choices=('default', 'full'), required=True)
    p.add_argument('--devices', required=True)
    p.add_argument('--models', type=Path, default=Path('/data1/models'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('decode', 'serving', 'fidelity', 'all'), default='all')
    p.add_argument('--port', type=int, default=8277)
    p.add_argument('--reuse-first-control', action='store_true',
                   help='Reuse an already completed five-trial control on the same profile/devices')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    profile = a.precision_profile
    arm = 'full_gdn' if profile == 'full' else 'gdn'
    base = dict(os.environ, CUDA_VISIBLE_DEVICES=a.devices, MXFP6_AUTOTUNE='off',
                VLLM_MACH_GDN_STRIDED_BA='0', VLLM_MACH_GDN_RECURRENT_TILE='32',
                VLLM_MACH_FUSED_GDN_QUANT='auto', VLLM_MACH_FUSED_SWIGLU_QUANT='auto')

    def run(label, variant, command):
        name = f'{profile}-{label}'
        marker = a.output / f'{name}.complete.json'
        empty, gate = VARIANTS[variant]
        env = dict(base, VLLM_MACH_GDN_EMPTY_OUTPUT=empty, VLLM_MACH_FUSED_ATTN_QUANT=gate)
        launch = dict(command=[sys.executable, *command], precision_profile=profile,
            variant=variant, environment={k:v for k,v in env.items()
                if k.startswith(('VLLM_', 'MXFP6_', 'CUDA_VISIBLE', 'PYTHONPATH'))})
        if marker.exists():
            assert json.loads(marker.read_text()) == launch, f'Cannot resume different contract: {name}'
            print('SKIP', name, flush=True)
            return
        (a.output / f'{name}.launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        print('START', name, flush=True)
        with (a.output / f'{name}.log').open('w') as log:
            subprocess.run(launch['command'], cwd=root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps(launch, indent=2) + '\n')
        print('DONE', name, flush=True)

    if a.phase in ('decode', 'all'):
        for block, order in enumerate(ORDERS):
            sizes = [4, 16, 24, 32] if block == 0 else [32, 24, 16, 4]
            for variant in order:
                label = f'r{block}-{variant}'
                if block == 0 and variant == 'control' and a.reuse_first_control:
                    directory = a.output / f'{profile}-{label}'
                    contract = json.loads((directory / 'contract.json').read_text())
                    assert contract['precision_profile'] == profile and contract['devices'] == a.devices
                    assert contract['config']['model'] == str(a.models / 'Qwen3.8-27B-MXFP6')
                    assert not contract['profile'] and contract['repeats'] == 5
                    assert contract['gdn_empty_output'] == contract['fused_attention_quant'] == '0'
                    rows = json.loads((directory / 'summary.json').read_text())
                    assert [r['rows'] for r in rows] == sizes
                    assert all(len(r['trials']) == 5 for r in rows)
                    print('REUSE', f'{profile}-{label}', flush=True)
                    continue
                run(label, variant, ['tools/benchmark_tp2_decode.py', '--model',
                    str(a.models / 'Qwen3.8-27B-MXFP6'), '--precision-profile', profile,
                    '--output', str(a.output / f'{profile}-{label}'),
                    '--rows', *map(str, sizes)])
    if a.phase in ('serving', 'all'):
        # Reverse the variant order across the two independent precision profiles.
        for variant in ORDERS[int(profile == 'full')]:
            label = f'serving-{variant}'
            run(label, variant, ['tools/compare_native_serving.py', '--arms', arm,
                '--models', str(a.models), '--stock-runtime', '/tmp/unused-stock',
                '--output', str(a.output / f'{profile}-{label}'), '--devices', a.devices,
                '--port', str(a.port), '--prompt-manifest', 'docs/data/serving-prompts.json'])
    if a.phase in ('fidelity', 'all'):
        for variant in ('control', 'both'):
            for m in (4, 32):
                label = f'fidelity-{variant}-m{m}'
                run(label, variant, ['tools/fidelity_native_mxfp6.py', '--arm', arm,
                    '--model', str(a.models / 'Qwen3.8-27B-MXFP6'),
                    '--tokenizer', str(a.models / 'Qwen3.8-27B-official'),
                    '--manifest', 'docs/data/fidelity-samples.json', '--physical-rows', str(m),
                    '--output', str(a.output / f'{profile}-{label}')])


if __name__ == '__main__':
    main()
