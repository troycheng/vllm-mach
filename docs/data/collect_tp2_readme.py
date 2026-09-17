"""Build all README comparison data from validated September 17 TP2 results."""
import argparse
import json
from pathlib import Path
import statistics

from collect_tp2_combinations import read, sha

HERE = Path(__file__).resolve().parent
ROWS = (4,16,24,32)


def collect(combinations, cumulative):
    combo = read(combinations)['profiles']
    full = read(cumulative)['variants']['current']
    stock_serving_path = HERE / 'native-serving-20260915.json'
    stock_fidelity_path = HERE / 'native-fidelity-20260915.json'
    stock_serving = read(stock_serving_path)
    stock_fidelity = read(stock_fidelity_path)
    bf16_reference_path = HERE / 'native-fidelity.json'
    assert stock_fidelity['queries'] == read(bf16_reference_path)['queries']
    performance, accuracy = {}, {}
    for arm in ('fp8', 'nvfp4'):
        performance[arm] = dict(source=stock_serving_path.name, repeats=1,
            points=[dict(concurrency=p['concurrency'],
                         output_throughput_tokens_per_s=p['output_throughput_tokens_per_s'],
                         replicate_rates=[p['output_throughput_tokens_per_s']])
                    for p in stock_serving['runs'][arm]['points']])
        f = stock_fidelity['runs'][arm]
        accuracy[arm] = {k:f[k] for k in ('mae','ci95','query_mae')}
        accuracy[arm]['source'] = stock_fidelity_path.name

    for arm, groups in (
        ('gdn', [combo['default']['serving']['control']]),
        ('full_gdn', full['serving']),
    ):
        assert len(groups) == (2 if arm == 'full_gdn' else 1)
        points = []
        for m in ROWS:
            runs = [next(p for p in g['points'] if p['concurrency'] == m) for g in groups]
            assert all(p['contract'] == stock_serving['contracts'][str(m)] for p in runs)
            rates = [p['aggregate']['output_throughput_tokens_per_s'] for p in runs]
            points.append(dict(concurrency=m, output_throughput_tokens_per_s=statistics.mean(rates),
                replicate_rates=rates, completed_requests=sum(p['aggregate']['completed'] for p in runs)))
        performance[arm] = dict(repeats=len(groups), points=points,
            source=combinations.name if arm == 'gdn' else cumulative.name)
        fidelity = combo['default']['fidelity']['32']['control'] if arm == 'gdn' else full['fidelity']['32']
        accuracy[arm] = {k:fidelity[k] for k in ('mae','ci95','records_sha256','contract')}

    head = full['fidelity']['32']['head']
    assert {r['rank'] for r in head} == {0,1}
    assert head[0]['rows'] == head[1]['rows'] > 0
    assert head[0]['global_argmax_mismatches'] == head[1]['global_argmax_mismatches']
    rows = head[0]['rows']
    sources = {p.name:sha(p) for p in (combinations,cumulative,stock_serving_path,stock_fidelity_path,bf16_reference_path,HERE / 'gdn-m4-fidelity.json')}
    return dict(schema='mach-tp2-readme/v1', sources=sources,
        accuracy=dict(runs=accuracy), performance=dict(runs=performance),
        fidelity_m4={arm:{k:f[k] for k in ('mae','ci95','records_sha256','contract')}
            for arm,f in [('gdn',combo['default']['fidelity']['4']['control']),('full_gdn',full['fidelity']['4'])]},
        head_probe=dict(eligible_rows=rows,
            global_top20_recall=1-sum(r['global_top20_candidate_misses_on_rank'] for r in head)/(20*rows),
            final_top1_agreement=1-head[0]['global_argmax_mismatches']/rows),
        scope='Sep 17 accepted configuration: default one serving run, full mean of two runs; stock Sep 15 reference data. No throughput confidence interval.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combinations', type=Path, default=HERE / 'tp2-combinations.json')
    p.add_argument('--cumulative', type=Path, default=HERE / 'tp2-full-cumulative.json')
    p.add_argument('--output', type=Path, default=HERE / 'readme-tp2-20260917.json')
    a = p.parse_args()
    a.output.write_text(json.dumps(collect(a.combinations,a.cumulative), indent=2) + '\n')
