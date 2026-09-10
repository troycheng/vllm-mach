"""Recompute MAE from the public gold-token logprobs; standard library only."""
import json
import math
from pathlib import Path
from statistics import mean

def validate(data):
    expected = {'fp8_accelerated028','fp8_stock029','mxfp6_champion',
                'k5k6_hybrid_fp16_ba','k4k5_derived_w6_fp16','nvfp4'}
    assert set(data['runs']) == expected
    queries = data['queries']
    assert len(queries) == len({q['id'] for q in queries}) == data['query_count'] == 256
    assert sum(q['target_tokens'] for q in queries) == data['target_tokens'] == 10479
    assert data['physical_rows'] == 32
    for name, run in data['runs'].items():
        assert len(run['gold_logprobs']) == len(run['query_mae']) == 256
        values = []
        for q, actual, recorded in zip(queries,run['gold_logprobs'],run['query_mae'],strict=True):
            reference = q['bf16_logprobs']
            assert len(actual) == len(reference) == q['target_tokens']
            assert all(math.isfinite(v) for v in actual+reference)
            mae = mean(abs(a-b) for a,b in zip(actual,reference,strict=True))
            assert math.isclose(mae,recorded,rel_tol=1e-10,abs_tol=1e-12)
            values.append(mae)
        assert math.isclose(mean(values),run['mae'],rel_tol=1e-10,abs_tol=1e-12)
        assert 0 <= run['ci95'][0] <= run['mae'] <= run['ci95'][1]
        print(f"{name}: MAE {run['mae']:.9f}, 256 queries verified")
    return data

if __name__ == '__main__':
    from check_quantization_comparison import LABELS
    here = Path(__file__).resolve().parent
    data = validate(json.loads((here/'accuracy-comparison-m32-20260910.json').read_text()))
    report = (here.parent/'benchmarks.md').read_text()
    for name, run in data['runs'].items():
        row = f"| {LABELS[name]} | {run['mae']:.6f} | {run['ci95'][0]:.6f}–{run['ci95'][1]:.6f} |"
        assert row in report, ('Accuracy table differs from evidence', row)
    print('Accuracy table matches the checked data')
