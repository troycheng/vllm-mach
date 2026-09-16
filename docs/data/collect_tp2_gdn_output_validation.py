"""Verify both-rank GDN output initialization and changing-size graph runs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_contract(root, control, candidate):
    base = read(root / control / 'contract.json')
    probe = read(root / candidate / 'contract.json')
    assert base.pop('gdn_empty_output') == '0'
    assert probe.pop('gdn_empty_output') == '1'
    assert base == probe
    return base


def trials(root, mode, expected_rows, empty_output):
    summaries = read(root / mode / 'summary.json')
    assert [r['rows'] for r in summaries] == expected_rows
    results = []
    for summary in summaries:
        rows = summary['rows']
        assert len(summary['trials']) == 5
        hashes = []
        for trial in range(5):
            path = root / f'{mode}/m{rows}-trial{trial}/result.json'
            result = read(path)
            assert len(result['metrics']) == rows
            assert all(v['num_generation_tokens'] == (129 if rows == 1 else 1025)
                       and not v['is_corrupted'] for v in result['metrics'])
            assert {r['rank'] for r in result['ranks']} == {0, 1}
            for rank in result['ranks']:
                assert rank['empty_output_layers'] == (48 if empty_output else 0)
                assert any(not b['has_prefill'] and
                           b['logical_rows'] == b['padded_rows'] == rows
                           for b in rank['batches'])
            hashes.append(sha(path))
        results.append({'mode': mode, 'rows': rows, 'raw_sha256': hashes,
            'mean_output_tokens_per_s': summary['mean_output_tokens_per_s'],
            'stdev_output_tokens_per_s': summary['stdev_output_tokens_per_s']})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name(
        'tp2-gdn-output-initialization-validation.json'))
    args = parser.parse_args()
    root = args.root
    for suffix in ('', '-profile'):
        matched_contract(root, f'off{suffix}', f'on{suffix}')
    decode_trials = [row for mode in ('off', 'on')
                     for row in trials(root, mode, [1, 4, 16, 32], mode == 'on')]
    traces = []
    for mode, empty_output in (('off', 0), ('on', 1)):
        for rows in (1, 4, 16, 32):
            directory = root / f'{mode}-profile/m{rows}-trial0'
            ranks = read(directory / 'result.json')['ranks']
            assert {r['rank'] for r in ranks} == {0, 1}
            for rank in ranks:
                assert len(rank['samples']) == 3
                assert all(s['logical_rows'] == s['padded_rows'] == rows
                           for s in rank['samples'])
                assert rank['empty_output_layers'] == (48 if empty_output else 0)
                path = directory / f'rank{rank["rank"]}.json'
                kernels = [e for e in read(path)['traceEvents']
                           if e.get('cat') == 'kernel']
                assert sum('allreduce_fusion' in e['name'] for e in kernels) == 384
                assert rank['gdn']['strided_ba_layers'] == 0
                fills = [e for e in kernels if 'FillFunctor<c10::BFloat16>' in e['name']]
                assert len(fills) == (0 if empty_output else 144), (mode, rows, len(fills))
                traces.append({'mode': mode, 'rows': rows, 'rank': rank['rank'],
                    'samples': 3, 'bf16_fills_per_decode': len(fills) // 3,
                    'bf16_fill_us_per_decode': sum(e['dur'] for e in fills) / 3,
                    'raw_path': str(path), 'sha256': sha(path)})
    regression_contract = matched_contract(root, 'regression-off', 'regression')
    regression = [row for mode in ('regression-off', 'regression')
                  for row in trials(root, mode, [2, 8, 24], mode == 'regression')]
    reverse_contract = matched_contract(root, 'reverse-off', 'reverse-on')
    reverse = [row for mode in ('reverse-on', 'reverse-off')
               for row in trials(root, mode, [32], mode == 'reverse-on')]
    assert '28 passed' in (root / 'output-tests.log').read_text()
    assert '60 passed' in (root / 'cpu-tests.log').read_text()
    for mode in ('off', 'on'):
        launch = read(root / f'serving-{mode}/gdn/launch.json')
        expected = '0' if mode == 'off' else '1'
        assert launch['environment'].pop('VLLM_MACH_GDN_EMPTY_OUTPUT') == expected
        if mode == 'off':
            baseline_launch = launch
        else:
            assert launch == baseline_launch
    wheel, = (root / 'wheel').glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        for name in ('gdn_decode.py', 'gdn/gdn_fused_decode_sm120.cu'):
            relative = 'vllm_mach/mxfp6/' + name
            assert archive.read(relative) == Path('src', relative).read_bytes()
    data = {'schema': 'mach-tp2-gdn-output-initialization-validation/v1', 'traces': traces,
        'wheel': {'path': str(wheel), 'sha256': sha(wheel)},
        'decode_trials': decode_trials,
        'reverse_order': {'contract': reverse_contract, 'runs': reverse},
        'sources': {str(p): sha(p) for p in (
            Path('src/vllm_mach/mxfp6/gdn/gdn_fused_decode_sm120.cu'),
            Path('src/vllm_mach/mxfp6/gdn_decode.py'),
            Path('tests/native_mxfp6/test_gdn_output_initialization.py'))},
        'tests': {'cases': 28, 'changing_input_replays_per_case': 120,
                   'state_dtypes': ['float32', 'float16'], 'convolution_layouts': ['SD', 'DS'],
                   'padding': [0, -1, 'all rows'],
                   'output_and_states_bitwise_equal': True,
                   'gpu_log_sha256': sha(root / 'output-tests.log'),
                   'cpu_log_sha256': sha(root / 'cpu-tests.log')},
        'regression': {'contract': regression_contract, 'rows': regression}}
    args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
