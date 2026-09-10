"""Validate the public request-level extract; standard library only."""
import json
import math
from pathlib import Path
from statistics import mean


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (actual, expected)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    return values[lower] + (values[math.ceil(pos)] - values[lower]) * (pos - lower)


def validate(data):
    total = 0
    assert len(data['runs']) == 4
    assert len({r['id'] for r in data['runs']}) == 4
    for run in data['runs']:
        c, a = run['contract'], run['aggregate']
        rows = [dict(zip(data['request_columns'], r)) for r in run['requests']]
        assert len(rows) == c['num_prompts'] == a['requested'] == a['completed']
        assert {r['index'] for r in rows} == set(range(len(rows)))
        assert a['duration_s'] > 0
        for r in rows:
            assert r['success'] is True and r['http_status'] == 200
            assert r['prompt_tokens'] == c['input_tokens']
            assert r['completion_tokens'] == c['output_tokens'] > 1
            assert 0 <= r['ttft_s'] <= r['latency_s'] and r['tpot_s'] >= 0
            close(r['tpot_s'], (r['latency_s'] - r['ttft_s']) / (r['completion_tokens'] - 1))
        assert sum(r['prompt_tokens'] for r in rows) == a['prompt_tokens']
        assert sum(r['completion_tokens'] for r in rows) == a['completion_tokens']
        close(a['completion_tokens'] / a['duration_s'], a['output_throughput_tokens_per_s'])
        close(len(rows) / a['duration_s'], a['request_throughput_per_s'])
        close((a['prompt_tokens'] + a['completion_tokens']) / a['duration_s'], a['total_throughput_tokens_per_s'])
        for field in ('ttft', 'tpot'):
            values = [r[field + '_s'] * 1000 for r in rows]
            close(mean(values), a['mean_' + field + '_ms'])
            close(percentile(values, 0.99), a['p99_' + field + '_ms'])
        total += len(rows)
        print(f"{run['id']}: {len(rows)} requests; throughput and latency verified")
    source = data['experimental_source']
    assert [p['concurrency'] for p in source['points']] == [4, 16, 24, 32]
    for point in source['points']:
        a = point['aggregate']
        assert a['completed'] == a['requested']
        assert a['prompt_tokens'] == a['completed'] * source['input_tokens']
        assert a['completion_tokens'] == a['completed'] * source['output_tokens']
        close(a['completion_tokens'] / a['duration_s'], a['output_throughput_tokens_per_s'])
    assert total == 640
    print('640 Mach request rows verified; experimental source aggregates checked separately')


if __name__ == '__main__':
    data = json.loads(Path(__file__).with_name('serving-results-20260909.json').read_text())
    validate(data)
    report = Path(__file__).resolve().parents[1] / 'benchmarks.md'
    text = report.read_text()
    for run in data['runs']:
        c, a = run['contract'], run['aggregate']
        row = (f"| {c['input_tokens']} / {c['output_tokens']} | {c['max_concurrency']} | "
               f"{c['num_prompts']} | {a['duration_s']:.2f} | {a['output_throughput_tokens_per_s']:.2f} | "
               f"{a['mean_tpot_ms']:.2f} | {a['p99_ttft_ms']:.2f} |")
        assert row in text, ('Release table differs from data', row)
    print('Release table matches the checked data')
