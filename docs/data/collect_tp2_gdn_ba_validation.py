"""Validate the strided BA probe's two-rank traces and graph regression."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from summarize_tp2_decode import summarize_trace


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name(
        'tp2-gdn-strided-ba-validation.json'))
    args = parser.parse_args()
    root = args.root
    traces = []
    for mode in ('off', 'on'):
        for rows in (1, 4, 16, 32):
            directory = root / f'{mode}-profile/m{rows}-trial0'
            ranks = read(directory / 'result.json')['ranks']
            assert {r['rank'] for r in ranks} == {0, 1}
            for rank in ranks:
                assert len(rank['samples']) == 3
                assert all(s['logical_rows'] == s['padded_rows'] == rows
                           for s in rank['samples'])
                path = directory / f'rank{rank["rank"]}.json'
                categories = summarize_trace(path)['categories']
                copies = categories.get('strided_tensor_copy', {}).get('count', 0)
                assert copies == (288 if mode == 'off' and rows >= 16 else 0)
                assert categories['allreduce_residual_norm']['count'] == 384
                if rows >= 16:
                    assert categories['gdn_recurrence']['count'] == 144
                assert rank['gdn']['strided_ba_layers'] == (48 if mode == 'on' else 0)
                traces.append(dict(mode=mode, rows=rows, rank=rank['rank'], samples=3,
                    copies_per_decode=copies // 3, allreduce_residual_norm_per_decode=128,
                    strided_ba_layers=rank['gdn']['strided_ba_layers'],
                    raw_path=str(path), sha256=sha(path)))

    assert '12 passed' in (root / 'strided-tests.log').read_text()
    assert '60 passed' in (root / 'cpu-tests.log').read_text()
    data = dict(schema='mach-tp2-gdn-strided-ba-validation/v1', traces=traces,
        tests=dict(strided_ba=dict(cases=12, replays_per_case=120,
            state_dtypes=['float32', 'float16'], convolution_layouts=['SD', 'DS'],
            rows=[16, 24, 32], output_and_states_bitwise_equal=True,
            convolution_null_index=0, recurrent_null_indices=[0, -1],
            log_sha256=sha(root / 'strided-tests.log')),
            cpu=dict(passed=60, log_sha256=sha(root / 'cpu-tests.log'))),
        off_launch=read(root / 'off-launch.json'))
    wheel, = (root / 'wheel').glob('*.whl')
    data['wheel'] = dict(path=str(wheel), sha256=sha(wheel))

    regression = []
    summary = read(root / 'regression/summary.json')
    assert [row['rows'] for row in summary] == [2, 8, 24]
    for row in summary:
        rows, trials = row['rows'], []
        assert len(row['trials']) == 5
        for trial in range(5):
            path = root / f'regression/m{rows}-trial{trial}/result.json'
            result = read(path)
            assert len(result['metrics']) == rows
            assert all(v['num_generation_tokens'] == 1025 and not v['is_corrupted']
                       for v in result['metrics'])
            assert {r['rank'] for r in result['ranks']} == {0, 1}
            ranks = []
            for rank in result['ranks']:
                matching = [b for b in rank['batches'] if not b['has_prefill']
                            and b['logical_rows'] == b['padded_rows'] == rows]
                assert matching and rank['gdn']['strided_ba_layers'] == 48
                ranks.append(dict(rank=rank['rank'], target_decode_batches=len(matching),
                                  strided_ba_layers=48))
            trials.append(dict(ranks=ranks, raw_sha256=sha(path)))
        regression.append(dict(rows=rows, trials=trials))
    data['regression'] = dict(contract=read(root / 'regression/contract.json'),
                              rows=regression)
    args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
