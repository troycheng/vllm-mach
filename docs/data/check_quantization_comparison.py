"""Recompute the public 3k/1k comparison without CUDA or third-party packages."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

LABELS = {
    'fp8_accelerated028': 'FP8 · accelerated vLLM 0.28',
    'fp8_stock029': 'FP8 · official vLLM 0.29',
    'mxfp6_champion': 'MXFP6 Champion',
    'k4k5_derived_w6_fp16': 'K4/K5-derived W6 + FP16 SSM',
    'k5k6_hybrid_fp16_ba': 'K5/K6 Hybrid + FP16 SSM + BA',
    'nvfp4': 'NVFP4 · local calibration',
}


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7), (actual, expected)


def p99(values):
    values = sorted(values)
    position = (len(values)-1)*.99
    lo = math.floor(position)
    return values[lo] + (values[math.ceil(position)]-values[lo])*(position-lo)


def validate(data):
    assert set(data['runs']) == set(LABELS)
    for c, contract in data['contracts'].items():
        payload = {k: v for k, v in contract.items() if k != 'sha256'}
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(canonical).hexdigest() == contract['sha256']
        assert contract['max_concurrency'] == int(c)
        assert len(contract['prompt_sha256']) == contract['num_prompts']
    total = 0
    for name, run in data['runs'].items():
        assert [p['concurrency'] for p in run['points']] == [4,16,24,32]
        for point in run['points']:
            c = point['concurrency']
            contract = data['contracts'][str(c)]
            n = {4:192,16:512,24:672,32:768}[c]
            assert point['contract_sha256'] == contract['sha256']
            assert contract['num_prompts'] == point['requests'] == n
            assert contract['input_tokens'] == 3000 and contract['output_tokens'] == 1000
            rows = [dict(zip(data['request_columns'], r)) for r in point['request_rows']]
            assert len(rows) == n and {r['index'] for r in rows} == set(range(n))
            for r in rows:
                assert r['success'] is True and r['http_status'] == 200
                assert r['prompt_tokens'] == 3000 and r['completion_tokens'] == 1000
                assert 0 <= r['ttft_s'] <= r['latency_s'] and r['tpot_s'] >= 0
                close(r['tpot_s'], (r['latency_s']-r['ttft_s'])/999)
            assert point['prompt_tokens'] == sum(r['prompt_tokens'] for r in rows)
            assert point['completion_tokens'] == sum(r['completion_tokens'] for r in rows)
            assert point['duration_s'] >= 300
            close(point['output_throughput_tokens_per_s'], point['completion_tokens']/point['duration_s'])
            for metric in ('ttft','tpot'):
                values = [r[metric+'_s']*1000 for r in rows]
                close(mean(values), point['mean_'+metric+'_ms'])
                close(p99(values), point['p99_'+metric+'_ms'])
            total += n
        print(name + ': 2144 request rows and all four points verified')
    assert total == 12864
    return total


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    data = json.loads((here/'quantization-comparison-3k1k-20260910.json').read_text())
    assert hashlib.sha256((here/'benchmark_fixed_token_contract.py').read_bytes()).hexdigest() == data['benchmark_script_sha256']
    total = validate(data)
    readme = (here.parents[1]/'README.md').read_text()
    report = (here.parent/'benchmarks.md').read_text()
    for name, run in data['runs'].items():
        row = '| ' + LABELS[name] + ' | ' + ' | '.join(f"{p['output_throughput_tokens_per_s']:.2f}" for p in run['points']) + ' |'
        assert row in report, ('Throughput table differs from evidence', row)
        latency_row = '| ' + LABELS[name] + ' | ' + ' | '.join(f"{p['mean_tpot_ms']:.2f}" for p in run['points']) + ' |'
        assert latency_row in report, ('Latency table differs from evidence', latency_row)
    assert 'docs/images/throughput-comparison.png' in readme
    print(f'{total} scored requests verified; throughput and latency tables match the data')
