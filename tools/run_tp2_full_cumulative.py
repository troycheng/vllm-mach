#!/usr/bin/env python3
"""Compare archived P0 with accepted optimizations under the same full profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-source', type=Path, required=True)
    p.add_argument('--baseline-extension', type=Path, required=True)
    p.add_argument('--current-extension', type=Path, required=True)
    p.add_argument('--current-library', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', default='4,5')
    p.add_argument('--port', type=int, default=8279)
    p.add_argument('--models', type=Path, default=Path('/data1/models'))
    p.add_argument('--phase', choices=('all', 'decode', 'serving', 'fidelity'), default='all')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    baseline_library = a.baseline_extension / 'mxfp6/mxfp6_torch.so'
    expected = json.loads((root / 'docs/data/tp2-p0.json').read_text())['extension_library_sha256']
    assert hashlib.sha256(baseline_library.read_bytes()).hexdigest() == expected
    sources = {'p0': a.baseline_source.resolve(), 'current': root}
    extensions = {'p0': a.baseline_extension.resolve(), 'current': a.current_extension.resolve()}
    libraries = {'p0': baseline_library.resolve(), 'current': a.current_library.resolve()}

    def run(label, variant, command):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'MXFP6_', 'MACH_TP2_'))}
        env.update(CUDA_VISIBLE_DEVICES=a.devices, MXFP6_AUTOTUNE='off',
            MXFP6_LIBRARY_PATH=str(libraries[variant]),
            PYTHONPATH=os.pathsep.join(map(str, [sources[variant] / 'src', extensions[variant], a.runtime.resolve()])),
            VLLM_MACH_GDN_EMPTY_OUTPUT='0', VLLM_MACH_FUSED_ATTN_QUANT='0',
            VLLM_MACH_GDN_STRIDED_BA='0', VLLM_MACH_GDN_RECURRENT_TILE='32',
            VLLM_MACH_FUSED_GDN_QUANT='auto', VLLM_MACH_FUSED_SWIGLU_QUANT='auto')
        launch = dict(command=[sys.executable, *command], variant=variant,
            source_commit=subprocess.check_output(['git', '-C', str(sources[variant]), 'rev-parse', 'HEAD'], text=True).strip(),
            library_sha256=hashlib.sha256(libraries[variant].read_bytes()).hexdigest(),
            environment={k: v for k, v in env.items() if k.startswith(('VLLM_', 'MXFP6_', 'CUDA_VISIBLE', 'PYTHONPATH'))})
        marker = a.output / f'{label}.complete.json'
        if marker.exists():
            assert json.loads(marker.read_text()) == launch
            print('SKIP', label, flush=True)
            return
        (a.output / f'{label}.launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        print('START', label, flush=True)
        with (a.output / f'{label}.log').open('w') as log:
            subprocess.run(launch['command'], cwd=root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps(launch, indent=2) + '\n')
        print('DONE', label, flush=True)

    for phase in ('decode', 'serving'):
        if a.phase not in ('all', phase):
            continue
        for block, order in enumerate((('p0', 'current'), ('current', 'p0'))):
            sizes = [4, 16, 24, 32] if block == 0 else [32, 24, 16, 4]
            for variant in order:
                label = f'{phase}-r{block}-{variant}'
                if phase == 'decode':
                    command = ['tools/benchmark_tp2_decode.py', '--precision-profile', 'full',
                        '--model', str(a.models / 'Qwen3.8-27B-MXFP6'),
                        '--rows', *map(str, sizes), '--output', str(a.output / label)]
                else:
                    command = ['tools/compare_native_serving.py', '--arms', 'full_gdn',
                        '--models', str(a.models), '--stock-runtime', '/tmp/unused-stock',
                        '--devices', a.devices, '--port', str(a.port),
                        '--prompt-manifest', 'docs/data/serving-prompts.json',
                        '--concurrencies', *map(str, sizes), '--output', str(a.output / label)]
                run(label, variant, command)
    if a.phase in ('all', 'fidelity'):
        for variant in ('p0', 'current'):
            for rows in (4, 32):
                label = f'fidelity-{variant}-m{rows}'
                run(label, variant, ['tools/fidelity_native_mxfp6.py', '--arm', 'full_gdn',
                    '--model', str(a.models / 'Qwen3.8-27B-MXFP6'),
                    '--tokenizer', str(a.models / 'Qwen3.8-27B-official'),
                    '--manifest', 'docs/data/fidelity-samples.json', '--physical-rows', str(rows),
                    '--output', str(a.output / label)])


if __name__ == '__main__':
    main()
