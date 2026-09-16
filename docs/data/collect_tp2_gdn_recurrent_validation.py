"""Verify both-rank recurrence launch geometry and changing-size graph runs."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name(
        'tp2-gdn-recurrent-validation.json'))
    args = parser.parse_args()
    root = args.root
    for suffix in ('', '-profile'):
        control = read(root / f'off{suffix}/contract.json')
        candidate = read(root / f'on{suffix}/contract.json')
        assert control.pop('gdn_recurrent_tile') == '32'
        assert candidate.pop('gdn_recurrent_tile') == '8'
        assert control == candidate
    decode_trials = []
    for mode, tile in (('off', 32), ('on', 8)):
        for rows in (1, 4, 16, 32):
            for trial in range(5):
                path = root / f'{mode}/m{rows}-trial{trial}/result.json'
                result = read(path)
                assert len(result['metrics']) == rows
                assert all(v['num_generation_tokens'] == (129 if rows == 1 else 1025)
                           and not v['is_corrupted'] for v in result['metrics'])
                assert {r['rank'] for r in result['ranks']} == {0, 1}
                for rank in result['ranks']:
                    assert rank['recurrent_tile8_layers'] == (48 if tile == 8 else 0)
                    assert any(not b['has_prefill'] and
                               b['logical_rows'] == b['padded_rows'] == rows
                               for b in rank['batches'])
                decode_trials.append({'mode': mode, 'rows': rows, 'trial': trial,
                                          'raw_sha256': sha(path)})
    traces = []
    for mode, tile in (('off', 32), ('on', 8)):
        for rows in (1, 4, 16, 32):
            directory = root / f'{mode}-profile/m{rows}-trial0'
            ranks = read(directory / 'result.json')['ranks']
            assert {r['rank'] for r in ranks} == {0, 1}
            for rank in ranks:
                assert len(rank['samples']) == 3
                assert all(s['logical_rows'] == s['padded_rows'] == rows
                           for s in rank['samples'])
                assert rank['recurrent_tile8_layers'] == (48 if tile == 8 else 0)
                path = directory / f'rank{rank["rank"]}.json'
                kernels = [e for e in read(path)['traceEvents']
                           if e.get('cat') == 'kernel']
                assert sum('allreduce_fusion' in e['name'] for e in kernels) == 384
                assert rank['gdn']['strided_ba_layers'] == 0
                recurrence = [e for e in kernels
                    if e['name'] == 'fused_recurrent_gated_delta_rule_packed_decode_kernel']
                assert len(recurrence) == (144 if rows >= 16 else 0)
                for event in recurrence:
                    assert event['args']['grid'] == [128 // tile, rows * 24, 1]
                    assert event['args']['block'] == [32, 1, 1]
                traces.append({'mode': mode, 'rows': rows, 'rank': rank['rank'],
                    'samples': 3, 'recurrent_launches_per_decode': len(recurrence) // 3,
                    'grid': recurrence[0]['args']['grid'] if recurrence else None,
                    'registers_per_thread': sorted({e['args']['registers per thread'] for e in recurrence}),
                    'recurrence_us_per_decode': sum(e['dur'] for e in recurrence) / 3,
                    'raw_path': str(path), 'sha256': sha(path)})
    regression = []
    for row in read(root / 'regression/summary.json'):
        rows = row['rows']
        assert len(row['trials']) == 5
        trials = []
        for trial in range(5):
            path = root / f'regression/m{rows}-trial{trial}/result.json'
            result = read(path)
            assert len(result['metrics']) == rows
            assert all(v['num_generation_tokens'] == 1025 and not v['is_corrupted']
                       for v in result['metrics'])
            assert {r['rank'] for r in result['ranks']} == {0, 1}
            for rank in result['ranks']:
                assert rank['recurrent_tile8_layers'] == 48
                assert any(not b['has_prefill'] and
                           b['logical_rows'] == b['padded_rows'] == rows
                           for b in rank['batches'])
            trials.append({'raw_sha256': sha(path)})
        regression.append({'rows': rows, 'trials': trials})
    assert [r['rows'] for r in regression] == [2, 8, 24]
    assert '24 passed' in (root / 'recurrent-tests.log').read_text()
    assert '57 passed' in (root / 'cpu-tests.log').read_text()
    wheel, = (root / 'wheel').glob('*.whl')
    data = {'schema': 'mach-tp2-gdn-recurrent-validation/v1', 'traces': traces,
        'wheel': {'path': str(wheel), 'sha256': sha(wheel)},
        'decode_trials': decode_trials,
        'sources': {str(p): sha(p) for p in (
            Path('src/vllm_mach/mxfp6/gdn/recurrent.py'),
            Path('src/vllm_mach/mxfp6/gdn_decode.py'),
            Path('tests/native_mxfp6/test_gdn_recurrent.py'))},
        'tests': {'cases': 24, 'changing_input_replays_per_case': 120,
                   'state_dtypes': ['float32', 'float16'], 'convolution_layouts': ['SD', 'DS'],
                   'ba_layouts': ['contiguous', 'strided'],
                   'output_and_states_bitwise_equal': True,
                   'gpu_log_sha256': sha(root / 'recurrent-tests.log'),
                   'cpu_passed': 57, 'cpu_log_sha256': sha(root / 'cpu-tests.log')},
        'regression': {'contract': read(root / 'regression/contract.json'), 'rows': regression}}
    args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
