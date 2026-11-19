#!/usr/bin/env python3
"""Render model-level serving tables and plots from checked-in measurements."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import geometric_mean
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data'
SOURCE_REV = 'eaa0acd6f5700913d10a422f6d4075614b9fb6be'
SOURCE_BASE = f'https://github.com/troycheng/vllm-mach/blob/{SOURCE_REV}/docs/'
GUIDE_BASE = 'https://github.com/troycheng/vllm-mach/blob/main/docs/'
SOURCES = {}
CURRENT_SOURCES = {'mxfp8-benchmark.json', 'mxfp8-fidelity-20261020.json',
                   'mxfp8-fidelity-20261020-records.json.gz'}
# Blue brand accents and neutral references, with GitHub-compatible backgrounds.
CHART_THEME = {
    'background': ('#ffffff', '#0d1117'),
    'text': ('#111827', '#e6edf3'),
    'muted': ('#526071', '#9da7b3'),
    'axis': ('#8c98a8', '#6e7681'),
    'grid': ('#cfd7e3', '#30363d'),
    'fp8': ('#9ba6b7', '#778394'),
    'nvfp4': ('#526279', '#a2b0c3'),
    'default': ('#9bc4f7', '#284f7d'),
    'full': ('#007bff', '#379bff'),
    'emphasis': ('#0066cc', '#58a6ff'),
    'interval': ('#354152', '#c9d1d9'),
}


def themed_svg(svg):
    for role,(light,_) in CHART_THEME.items():
        svg=svg.replace(light, f'var(--chart-{role}, {light})')
    light=' '.join(f'--chart-{role}: {pair[0]};' for role,pair in CHART_THEME.items())
    dark=' '.join(f'--chart-{role}: {pair[1]};' for role,pair in CHART_THEME.items())
    css=f':root {{ {light} }} @media (prefers-color-scheme: dark) {{ :root {{ {dark} }} }}'
    return svg.replace('<style type="text/css">', '<style type="text/css">'+css+'\n', 1)


def load(name):
    path = DATA / name
    SOURCES[name] = dict(url=(GUIDE_BASE if name in CURRENT_SOURCES else SOURCE_BASE) + 'data/' + name,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return json.loads(path.read_text())


def point(c, baseline, selected, **extra):
    assert baseline > 0 and selected > 0
    return dict(c=int(c), baseline_tps=baseline, selected_tps=selected,
                gain_pct=100 * (selected / baseline - 1), **extra)


def dataset():
    mx = load('mxfp8-benchmark.json')
    assert mx['complete'], 'Complete the MXFP8 configuration matrix first'
    mx_quality = load('mxfp8-fidelity-20261020.json')
    assert mx_quality['complete']
    block = load('fp8-qualification-20261006.json')
    dense = load('dense-gains-aligned-baseline-20260920.json')
    core = load('dense-core-ablation-20260920.json')
    moe = load('moe-recheck-20260928.json')
    dense_nvfp4 = load('native-serving.json')
    moe_nvfp4 = load('qwen35-default-full-20260917.json')
    fidelity = load('profile-fidelity-20260917.json')
    def mx_model(model):
        variants = {v['id']: v for v in mx['models'][model]['variants']}
        selected = (variants['baseline'] if model == '2b' else
                    mx['dtype_attention_recheck']['variants']['fp32-dtype']['fp8_e4m3'])
        optional = variants['ssm_bf16' if model == '2b' else 'ssm_bf16_kv_fp8']
        refs = {p['concurrency']: p['output_tokens_s'] for p in mx['block_fp8_references'][model]['points']}
        kv = 'BF16' if model == '2b' else 'FP8'
        return dict(id='qwen35-'+model, title='Qwen3.5-'+model.upper(),
                    profile=f'MXFP8 · FP32 SSM / {kv} KV', tp=1, date=mx['date'],
                    guide='mxfp8-'+('2b-' if model == '2b' else '')+'champion.md',
                    source='mxfp8-benchmark.json', mxfp8=True, kv_dtype=kv,
                    scope='Greedy generation, October 14 configuration sweep; block-FP8 references are historical. BF16 SSM is optional and uses ordered GDN.',
                    points=[point(p['concurrency'], refs[p['concurrency']], p['metrics']['output_throughput_tokens_per_s'],
                                  bf16_tps=q['metrics']['output_throughput_tokens_per_s'],
                                  bf16_gain_pct=100*(q['metrics']['output_throughput_tokens_per_s']/refs[p['concurrency']]-1),
                                  bf16_vs_main_pct=100*(q['metrics']['output_throughput_tokens_per_s']/p['metrics']['output_throughput_tokens_per_s']-1))
                            for p,q in zip(selected['points'], optional['points'])])
    models = [mx_model('2b'), mx_model('4b'),
        dict(id='qwen35-4b-block-fp8', title='Qwen3.5-4B block-FP8', profile='Block-FP8 · all four features', tp=1,
             date='2026-10-06', guide='fp8-profile.md',
             scope='One same-stack stock/all-features pair at each point; 680 requests per arm. Its FP8 baseline is separate from the MXFP8 study.',
             source='fp8-qualification-20261006.json',
             points=[point(x['concurrency'], x['stock']['output_throughput_tokens_per_s'],
                           x['all_features']['output_throughput_tokens_per_s'],
                           mean_ttft_ms=x['all_features']['mean_ttft_ms'], mean_tpot_ms=x['all_features']['mean_tpot_ms'])
                     for x in block['benchmark']['points']]),
    ]
    rows = {x['arm']: x for x in dense['records']}
    fp8 = core['groups']['stock-path-aligned']['arms']['fp8']['points']
    dense_nv = {str(p['concurrency']): p['output_throughput_tokens_per_s']
                for p in dense_nvfp4['runs']['nvfp4']['points']}
    moe_nv = {str(p['concurrency']): p['nvfp4']['aggregate']['output_throughput_tokens_per_s']
              for p in moe_nvfp4['comparisons']}
    models.append(dict(id='qwen38-27b', title='Qwen3.8-27B Dense', profile='MXFP6 default / full', tp=2,
                       date='2026-09-20', guide='dense-gains-aligned-baseline-20260920.md',
                       scope='One configuration sweep; runtime, GPU-group and measurement-batch differences are recorded in the study.',
                       source='dense-gains-aligned-baseline-20260920.json',
                       nvfp4_source='native-serving.json', nvfp4_date='2026-09-15',
                       fidelity_key='dense',
                       points=[point(c, fp8[c]['output_throughput_tokens_per_s'], rows['dense_full']['throughput'][c],
                                     default_tps=rows['dense_default']['throughput'][c],
                                     nvfp4_tps=dense_nv[c],
                                     nvfp4_gain_pct=100*(rows['dense_full']['throughput'][c]/dense_nv[c]-1),
                                     default_nvfp4_gain_pct=100*(rows['dense_default']['throughput'][c]/dense_nv[c]-1),
                                     default_gain_pct=100*(rows['dense_default']['throughput'][c]/fp8[c]['output_throughput_tokens_per_s']-1))
                               for c in ('4','16','24','32')]))
    models.append(dict(id='qwen35-35b-a3b', title='Qwen3.5-35B-A3B MoE', profile='MXFP6 default / full', tp=2,
                       date='2026-09-28', guide='moe-gains-fp8-baseline-20260920.md',
                       scope='MXFP6 means of two sweeps versus September 17 FP8. The measured default has projection tuning off; full includes it. Current launchers enable eligible tuning.',
                       source='moe-recheck-20260928.json',
                       nvfp4_source='qwen35-default-full-20260917.json', nvfp4_date='2026-09-17',
                       fidelity_key='moe',
                       points=[point(c, moe['baseline'][c]['aggregate']['output_throughput_tokens_per_s'],
                                     moe['arms']['projection']['points'][c]['throughput'],
                                     default_tps=moe['arms']['default']['points'][c]['throughput'],
                                     nvfp4_tps=moe_nv[c],
                                     nvfp4_gain_pct=100*(moe['arms']['projection']['points'][c]['throughput']/moe_nv[c]-1),
                                     default_nvfp4_gain_pct=100*(moe['arms']['default']['points'][c]['throughput']/moe_nv[c]-1),
                                     default_gain_pct=100*(moe['arms']['default']['points'][c]['throughput']/moe['baseline'][c]['aggregate']['output_throughput_tokens_per_s']-1))
                               for c in ('4','16','24','32')]))
    for model in models:
        assert [p['c'] for p in model['points']] == sorted({p['c'] for p in model['points']})
        if model.get('mxfp8'):
            assert [p['c'] for p in model['points']] == [4, 16, 24, 32, 48, 64]
    # Preserve the published schema of the vLLM 0.29 archive.
    return dict(schema_version=3, updated=mx_quality['date'], unit='output tokens/s',
                workload=dict(input_tokens=3000, output_tokens=1000), gpu='RTX 5090 32 GiB',
                full_profile='MXFP6 full adds FP16 recurrent state and NVFP4 candidate search with BF16 head refinement.',
                sources=SOURCES, models=models,
                fidelity=dict(date_native='2026-09-17',
                    date_mxfp8=mx_quality['date'],
                    mxfp8={name:dict(
                        attention=m['attention'], kv_dtype=m['kv_dtype'], anchor=m['anchor'],
                        points=[dict(rows=p['rows'], bf16_vs_fp32=p['bf16_vs_fp32'],
                            **{dtype:{k:p[dtype][k] for k in ('mae','ci95','repeat')}
                               for dtype in ('float32','bfloat16')}) for p in m['points']])
                        for name,m in mx_quality['models'].items()},
                    native={name:{arm:run['mae'] for arm,run in m['m32']['runs'].items()}
                            for name,m in fidelity['models'].items()},
                    native_ci95={name:{arm:run['ci95'] for arm,run in m['m32']['runs'].items()}
                                 for name,m in fidelity['models'].items()},
                    bootstrap=fidelity['bootstrap']))


def asset_name(data, model, kind):
    return f"{model['id']}{data.get('asset_suffix', '')}-{kind}.svg"


def runtime_caption(data):
    runtime = data.get('runtime', {})
    return f"vLLM {runtime['vllm']} / Torch {runtime['torch']}" if runtime else 'Historical runtime'


def migration_data(path):
    """Read a finalized normalized dataset; never fill unfinished measurements."""
    data = json.loads(path.read_text())
    def complete(value):
        if value is None or isinstance(value, str) and 'pending' in value.lower():
            raise ValueError('Migration data contains pending/null values')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Migration data contains non-finite values')
        if isinstance(value, dict):
            for item in value.values(): complete(item)
        elif isinstance(value, list):
            for item in value: complete(item)
    complete(data)
    if data.get('publication_status') != 'accepted':
        raise ValueError('Migration dataset is not accepted for publication')
    if not re.fullmatch(r'-[a-z0-9]+(?:-[a-z0-9]+)*', data.get('asset_suffix', '')):
        raise ValueError('Migration data requires a distinct safe asset suffix')
    if not all(isinstance(data.get('runtime', {}).get(k), str) and data['runtime'][k]
               for k in ('vllm', 'torch', 'triton')):
        raise ValueError('Migration runtime identity is incomplete')
    if data.get('workload') != dict(input_tokens=3000, output_tokens=1000):
        raise ValueError('Migration workload differs from the serving contract')
    if data.get('unit') != 'output tokens/s' or not data.get('models') or not data.get('sources'):
        raise ValueError('Migration model/source data is incomplete')
    ids = set()
    for model in data['models']:
        if model.get('configuration_status') != 'accepted':
            raise ValueError('Included configuration has not been accepted')
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', model['id']) or model['id'] in ids:
            raise ValueError('Invalid or duplicate model ID')
        ids.add(model['id'])
        for field in ('title', 'profile', 'date', 'guide', 'source'):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ValueError('Missing migration model identity: '+field)
        if model['source'] not in data['sources'] or type(model.get('tp')) is not int or model['tp'] < 1:
            raise ValueError('Missing source or TP identity')
        points = model.get('points', [])
        if not points or any(type(p.get('c')) is not int or p['c'] <= 0 for p in points):
            raise ValueError('Invalid concurrency inventory')
        if [p['c'] for p in points] != sorted({p['c'] for p in points}):
            raise ValueError('Concurrency points must be unique and ordered')
        for point in points:
            for field in ('baseline_tps', 'selected_tps'):
                if type(point.get(field)) not in (int, float) or point[field] <= 0:
                    raise ValueError('Missing positive throughput: '+field)
            gain = 100 * (point['selected_tps'] / point['baseline_tps'] - 1)
            if type(point.get('gain_pct')) not in (int, float) or not math.isclose(point['gain_pct'], gain, abs_tol=1e-8):
                raise ValueError('Migration throughput ratio differs')
            for _, field, _, _ in throughput_series(model):
                if type(point.get(field)) not in (int, float) or point[field] <= 0:
                    raise ValueError('Incomplete plotted series: '+field)
        if model.get('mxfp8'):
            quality = data['fidelity']['mxfp8'][model['id'].removeprefix('qwen35-')]
            if not quality.get('points') or any(dtype not in p for p in quality['points'] for dtype in ('float32', 'bfloat16')):
                raise ValueError('Incomplete MXFP8 precision series')
        if 'fidelity_key' in model:
            quality = data['fidelity']['native'][model['fidelity_key']]
            intervals = data['fidelity']['native_ci95'][model['fidelity_key']]
            if not all(k in quality and k in intervals for k in ('fp8', 'nvfp4', 'default', 'full')):
                raise ValueError('Incomplete native precision series')
    return data


def throughput_series(model):
    if 'series' in model:
        series = [tuple(row) for row in model['series']]
        if not series or any(len(row) != 4 or row[2] not in ('fp8', 'nvfp4', 'default', 'full') for row in series):
            raise ValueError('Invalid explicit throughput series')
        return series
    if 'nvfp4_tps' in model['points'][0]:
        return [('FP8 baseline', 'baseline_tps', 'fp8', ''), ('NVFP4 reference', 'nvfp4_tps', 'nvfp4', '//'),
                ('MXFP6 default', 'default_tps', 'default', ''), ('MXFP6 full', 'selected_tps', 'full', '')]
    if model.get('mxfp8'):
        return [('Historical block FP8', 'baseline_tps', 'fp8', ''),
                (f"MXFP8 FP32 SSM / {model['kv_dtype']} KV", 'selected_tps', 'full', ''),
                (f"Optional BF16 SSM / {model['kv_dtype']} KV", 'bf16_tps', 'default', '')]
    return [('FP8 baseline', 'baseline_tps', 'fp8', ''),
            ('Mach block-FP8' if model['id']=='qwen35-4b-block-fp8' else 'MXFP8 champion', 'selected_tps', 'full', '')]


def vllm034_table():
    qualification = json.loads((DATA / 'vllm034-qualification.json').read_text())
    final_2b = json.loads((DATA / 'vllm034-2b-followup.json').read_text())
    final_by_concurrency = {row['concurrency']: row['summary'] for row in final_2b['serving']}
    lines = ['2B and 4B run on one GPU. The 27B and 35B models use two GPUs (TP2).', '',
             '| Model / profile | SSM | Concurrency | vLLM 0.33 | vLLM 0.34 | Change |',
             '|---|---|---:|---:|---:|---:|']
    for row in qualification['serving']:
        concurrency = row['concurrency']
        if row['profile'] == final_2b['profile']:
            final = final_by_concurrency[concurrency]
            before = final['0.33']['output_tok_s']
            after = final['0.34-final']['output_tok_s']
        else:
            before = geometric_mean(row['0.33_output_tok_s'])
            after = geometric_mean(row['0.34_output_tok_s'])
        state = {'float32': 'FP32', 'float16': 'FP16'}[row['ssm_dtype']]
        change = 100 * (after / before - 1)
        lines.append(f"| {row['profile']} | {state} | c{concurrency} | {before:,.2f} | {after:,.2f} | {change:+.2f}% |")
    return '\n'.join(lines)


def migration_markdown(data, native=False, release_table=None):
    lines = ['# Model serving performance', '',
             'RTX 5090 · 3,000 input / 1,000 output tokens · output tokens/s.', '']
    if release_table:
        base = 'https://github.com/troycheng/vllm-mach/blob/v0.4.0/docs/' if native else ''
        lines += ['## vLLM 0.34 throughput', '', release_table, '',
                  f'[Precision and KV capacity]({base}installation.md#vllm-034-qualification) · '
                  f'[Final 2B measurements]({base}data/vllm034-2b-followup.json) · '
                  f'[Migration measurements]({base}data/vllm034-qualification.json) · '
                  f'[27B precision with 400-token blocks]({base}data/vllm034-dense-400-gold.json)', '']
    lines += [f"## vLLM {data['runtime']['vllm']} concurrency curves", '',
             f"These runs use PyTorch {data['runtime']['torch']}. FP32 recurrent state is the default.", '',
             profile_overview(data, 'performance.md'), '']
    for model in data['models']:
        series = throughput_series(model)
        for alias in model.get('anchor_aliases', []):
            lines += [f'<a id="{alias}"></a>', '']
        gain_label = f"{model.get('selected_label', 'Mach')} gain vs {model.get('gain_reference_label', 'FP8')}"
        lines += [f'<a id="{model["id"]}"></a>', '', f"## {model.get('display_name', model['title'])}", '',
                  f"{model['profile']} · TP{model['tp']} · {model['date']}", '',
                  f"![{model['title']} throughput](assets/{asset_name(data, model, 'throughput')})", '',
                  '| Concurrency | '+' | '.join(row[0] for row in series)+f' | {gain_label} |',
                  '|---|'+ '|'.join('---:' for _ in series)+'|---:|']
        for point in model['points']:
            lines.append(f"| c{point['c']} | "+' | '.join(f"{point[row[1]]:.2f}" for row in series)+f" | {point['gain_pct']:+.2f}% |")
        if model.get('mxfp8') or 'fidelity_key' in model:
            lines += ['', '### Precision', '', f"![{model['title']} precision](assets/{asset_name(data, model, 'precision')})"]
        if model.get('precision_note'):
            lines += ['', '### Precision', '', model['precision_note']]
        if model.get('precision_comparison'):
            quality = data['fidelity'][model['precision_comparison']]
            lines += ['', f"![{model['title']} precision](assets/{asset_name(data, model, 'precision')})", '',
                      f"| Decode rows | {quality['reference_label']} MAE | Mach MAE |", '|---|---:|---:|']
            for point in quality['points']:
                lines.append(f"| M{point['M']} | {point['reference_mae']:.6f} | {point['mach_mae']:.6f} |")
        if model.get('scope'):
            lines += ['', model['scope']]
        links = [f"[Setup and benchmark]({GUIDE_BASE}{model['guide']})",
                 f"[Measurements]({data['sources'][model['source']]['url']})"]
        linked = {model['source']}
        for label, field, _, _ in series:
            source_id = model.get('series_sources', {}).get(field)
            if source_id and source_id not in linked:
                links.append(f"[{label} measurements]({data['sources'][source_id]['url']})")
                linked.add(source_id)
        lines += ['', ' · '.join(links), '']
    lines += ['## Measurement records', '',
              f"[Data and measurement protocol]({data['evidence_protocol']}) · [Normalized data](data/model-performance-vllm033.json) · [vLLM 0.29 → 0.33 comparison](data/evidence-vllm033/migration-status.json)", '',
              f"[Historical vLLM 0.29 FP8/NVFP4 comparisons]({GUIDE_BASE}performance-vllm029.md)", '']
    return '\n'.join(lines)


def profile_overview(data, report):
    lines = ['| Model / profile | GPUs | Measured concurrency |', '|---|---:|---|']
    for model in data['models']:
        points = ', '.join(f"c{point['c']}" for point in model['points'])
        lines.append(f"| [{model.get('display_name', model['title'])}]({report}#{model['id']}) | {model['tp']} | {points} |")
    return '\n'.join(lines)


def readme_performance(data, native=False, release_table=None):
    lines = ['## Serving performance', '',
             'RTX 5090 · 3,000 input / 1,000 output tokens · output tokens/s.', '']
    if release_table:
        lines += ['### vLLM 0.34 throughput', '', release_table, '',
                  '[Measurements and precision](docs/performance.md#vllm-034-throughput)', '']
    lines += [f"### vLLM {data['runtime']['vllm']} concurrency curves", '',
             'FP32 recurrent state is the default. The performance report includes optional BF16/FP16 state configurations and precision against BF16.', '']
    for model in data['models']:
        guide = GUIDE_BASE + model['guide'] if native else 'docs/' + model['guide']
        lines += [f"#### {model.get('display_name', model['title'])}", '',
                  f"![{model['title']} throughput](docs/assets/{asset_name(data, model, 'throughput')})", '',
                  f"[Data and precision](docs/performance.md#{model['id']}) · [Run this profile]({guide})", '']
    historical = GUIDE_BASE + 'performance-vllm029.md' if native else 'docs/performance-vllm029.md'
    lines += [f"[Historical vLLM 0.29 FP8/NVFP4 comparisons]({historical})", '']
    return '\n'.join(lines)


def update_readme(text, section):
    begin, end = '<!-- PERFORMANCE:BEGIN -->', '<!-- PERFORMANCE:END -->'
    before, middle = text.split(begin, 1)
    _, after = middle.split(end, 1)
    return before + begin + '\n' + section + '\n' + end + after


def overview(data):
    lines = ['| Model / profile | GPUs | Throughput gain vs FP8 | Details |',
             '|---|---|---:|---|']
    for m in data['models']:
        gains = [p['gain_pct'] for p in m['points']]
        label = f"{min(gains):+.2f}% to {max(gains):+.2f}%"
        if data.get('asset_suffix'):
            summaries = []
            for title, field in m.get('overview_series', [(m.get('selected_label', ''), 'selected_tps')]):
                values = [100 * (p[field] / p['baseline_tps'] - 1) for p in m['points']]
                summaries.append(f"{title} {min(values):+.2f}% to {max(values):+.2f}%".strip())
            label = '; '.join(summaries)
        elif m.get('mxfp8'):
            bg = [p['bf16_gain_pct'] for p in m['points']]
            label = f"FP32 {label}; BF16 optional {min(bg):+.2f}% to {max(bg):+.2f}%"
        elif 'default_tps' in m['points'][0] and not data.get('asset_suffix'):
            dg = [p['default_gain_pct'] for p in m['points']]
            label = f"default {min(dg):+.2f}% to {max(dg):+.2f}%; full {label}"
        lines.append(f"| {m['title']} · {m['profile']} | {m['tp']} · TP{m['tp']} | {label} | [Data](docs/performance.md#{m['id']}) |")
    return '\n'.join(lines)


def markdown(data, native=False):
    def guide(path):
        if native:
            return (SOURCE_BASE if path.startswith('data/') and Path(path).name not in CURRENT_SOURCES
                    and Path(path).name != 'fidelity-samples.json' else GUIDE_BASE) + path
        return path
    q=data['fidelity']; dense_q=q['native']['dense']; moe_q=q['native']['moe']
    lines = ['# Historical model serving performance · vLLM 0.29', '',
             'Historical vLLM 0.29 / Torch 2.13 configurations on RTX 5090: **3,000 input / 1,000 output tokens**, text-only, non-speculative inference. Throughput is **output tokens/s**; `c` is concurrent requests, and physical kernel rows are denoted `M`.', '',
             'Each model has its own throughput chart. The 27B Dense and 35B-A3B MoE comparisons include FP8, NVFP4 and both MXFP6 profiles, followed by a separate precision chart against BF16.', '',
             'MXFP6 **default** retains FP32 recurrent state and the BF16 head. **Full** adds FP16 state and NVFP4 candidate search with BF16 refinement. MXFP8 uses FP32 SSM by default: BF16 KV on 2B and FP8 KV on 4B. BF16 SSM is an optional configuration.', '']
    for m in sorted(data['models'], key=lambda m: m['id']=='qwen35-4b-block-fp8'):
        lines += [f"<a id=\"{m['id']}\"></a>", f"## {m['title']} — {m['profile']}", '',
                  f"Mach profile: {m['date']} · {m['tp']} × RTX 5090 · TP{m['tp']}", '',
                  f"![{m['title']} output throughput by concurrency](assets/{m['id']}-throughput.svg)", '']
        if 'default_tps' in m['points'][0]:
            lines += ['| Concurrency | FP8 | NVFP4 | MXFP6 default | MXFP6 full |', '|---|---:|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['baseline_tps']:.2f} | {p['nvfp4_tps']:.2f} | {p['default_tps']:.2f} | {p['selected_tps']:.2f} |")
            lines += ['', '| Concurrency | Default vs FP8 | Full vs FP8 | Default vs NVFP4 | Full vs NVFP4 |',
                      '|---|---:|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['default_gain_pct']:+.2f}% | {p['gain_pct']:+.2f}% | {p['default_nvfp4_gain_pct']:+.2f}% | {p['nvfp4_gain_pct']:+.2f}% |")
            key=m['fidelity_key']; quality=q['native'][key]
            lines += ['', f"### {m['title']} precision", '',
                      f"![{m['title']} gold-token logprob MAE against BF16](assets/{m['id']}-precision.svg)", '',
                      '| FP8 MAE | NVFP4 MAE | MXFP6 default MAE | MXFP6 full MAE |',
                      '|---:|---:|---:|---:|',
                      f"| {quality['fp8']:.5f} | {quality['nvfp4']:.5f} | {quality['default']:.5f} | {quality['full']:.5f} |", '',
                      'M32 · 256 queries / 10,479 fixed gold tokens · query-mean logprob MAE against BF16. Lower is better. Error bars show 95% query-bootstrap intervals.', '',
                      f"[NVFP4 throughput ({m['nvfp4_date']})]({guide('data/'+m['nvfp4_source'])}) · [Precision measurements]({guide('data/profile-fidelity-20260917.json')})"]
        elif m.get('mxfp8'):
            lines += ['| Concurrency | Historical block FP8 | MXFP8 FP32 SSM | vs block FP8 | Optional BF16 SSM | vs block FP8 | BF16 vs FP32 |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['baseline_tps']:.2f} | {p['selected_tps']:.2f} | {p['gain_pct']:+.2f}% | {p['bf16_tps']:.2f} | {p['bf16_gain_pct']:+.2f}% | {p['bf16_vs_main_pct']:+.2f}% |")
            lines += ['', f"Both MXFP8 curves use {m['kv_dtype']} KV and ordered GDN. FP32 SSM is the launcher default; add `--ssm-dtype bfloat16` for the optional curve. "
                      'Block-FP8 rates are historical references; MXFP8 curves are the October 14 greedy configuration sweep. '
                      'These ratios compare complete configurations.', '',
                      f"[October 15 random-sampling and best-path recheck]({guide('best-path-recheck-20261015.md')}) uses a separate sampling workload."]
            quality=q['mxfp8'][m['id'].removeprefix('qwen35-')]
            lines += ['', f"### {m['title']} precision", '',
                      f"![{m['title']} FP32 and BF16 SSM error against BF16](assets/{m['id']}-precision.svg)", '',
                      '| Physical rows | FP32 SSM MAE [95% CI] | Optional BF16 SSM MAE [95% CI] | BF16 vs FP32 |',
                      '|---|---:|---:|---:|']
            for p in quality['points']:
                def cell(dtype):
                    a=p[dtype]
                    return f"{a['mae']:.5f} [{a['ci95'][0]:.5f}, {a['ci95'][1]:.5f}]"
                lines.append(f"| M{p['rows']} | {cell('float32')} | {cell('bfloat16')} | {p['bf16_vs_fp32']['relative_pct']:+.2f}% |")
            lines += ['', 'October 20 · 256 queries / 10,479 fixed gold tokens per configuration · query-mean logprob MAE against BF16. Lower is better; intervals use 20,000 query-bootstrap samples. '
                      f"{quality['anchor']}. "
                      'All first-cohort repeats were identical.', '',
                      f"[Precision data]({guide('data/mxfp8-fidelity-20261020.json')}) · "
                      f"[Raw logprobs and receipts]({guide('data/mxfp8-fidelity-20261020-records.json.gz')}) · "
                      f"[Configuration and reproduction]({guide(m['guide'])})"]
        else:
            lines += ['| Concurrency | FP8 baseline | Selected profile | Gain |', '|---|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['baseline_tps']:.2f} | {p['selected_tps']:.2f} | {p['gain_pct']:+.2f}% |")
        if 'mean_ttft_ms' in m['points'][0]:
            lines += ['', '| Concurrency | Selected mean TTFT (ms) | Selected mean TPOT (ms) |', '|---|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['mean_ttft_ms']:.2f} | {p['mean_tpot_ms']:.3f} |")
        lines += ['', f"[Profile / study]({guide(m['guide'])}) · [Source measurements]({guide('data/'+m['source'])})", '']
    lines += ['## Numerical fidelity', '',
              'The September 17 MXFP6 evaluation measures token-logprob MAE against BF16 using 256 queries / 10,479 tokens. Lower MAE means closer numerical agreement with BF16.', '',
              '| Profile | Physical rows | Numerical result |', '|---|---|---|',
              '| 2B / 4B MXFP8 | M32/M64 | October 20: eight current-runtime configurations, with FP32 and BF16 SSM reported separately above. |',
              '| 4B block-FP8 | M4/32/64 | Fixed-gold records byte-identical to stock block-FP8. |',
              f"| 27B Dense MXFP6 default / full | M32 | BF16 MAE {dense_q['default']:.5f} / {dense_q['full']:.5f}; FP8 {dense_q['fp8']:.5f}, NVFP4 {dense_q['nvfp4']:.5f}. |",
              f"| 35B-A3B MoE MXFP6 default / full | M32 | BF16 MAE {moe_q['default']:.5f} / {moe_q['full']:.5f}; FP8 {moe_q['fp8']:.5f}, NVFP4 {moe_q['nvfp4']:.5f}. |", '',
              f"[2B precision and generation coverage]({guide('mxfp8-2b-champion.md#precision')}) · [4B qualification]({guide('mxfp8-champion.md#validation')}) · [MXFP6 fidelity methodology]({guide('profile-fidelity-20260917.md')})", '',
              '### Measurement protocol', '',
              'The linked studies and [source records](data/model-performance.json) retain hardware identities, software revisions, request counts and baseline dates. '
              'MXFP8 precision uses Mach `afcb4cf` and native `ce0cd05`; MXFP6 precision uses the September 17 evaluation. '
              'Logprob checks use the complete head: BF16 for MXFP6/2B, and the existing NVFP4 head with K2048 BF16 score refinement for 4B. '
              f"The [frozen token manifest]({guide('data/fidelity-samples.json')}) is public. Hybrid candidate selection has separate generation checks in the profile guides.", '',
              'The 27B NVFP4 throughput reference is the September 15 token-ID sweep. The 35B-A3B NVFP4 throughput reference is the September 17 user service; its launch configuration was not recorded. The separate 35B-A3B precision evaluation used the NVFP4 ScaleSweep checkpoint. Source records preserve these identities and measurement dates.', '',
              '### Reproduce MXFP8 fidelity', '',
              'Run from a vLLM Mach checkout inside the pinned MXFP8 environment with a prepared checkpoint:', '',
              '```bash',
              'CUDA_VISIBLE_DEVICES=0 python tools/measure_mxfp8_fidelity.py \\',
              '  --profile 2b --model /models/PREPARED-2B --rows 32 \\',
              '  --ssm-dtype float32 --outdir /results/2b-m32-fp32',
              '```', '',
              'Use `--profile 4b` for the prepared 4B model. Run each M32/M64 × float32/bfloat16 combination in a separate process and fresh output directory. '
              'The tool saves raw gold logprobs, model/library identities, runtime receipts and repeat differences. '
              'The published compressed record bundle includes the BF16 anchors. For each query, average the absolute logprob differences over its gold tokens, then average those 256 query errors.', '',
              '[Validation tool](https://github.com/troycheng/vllm-mach/blob/main/tools/measure_mxfp8_fidelity.py) · '
              f"[Results and raw-record bundle]({guide('data/mxfp8-fidelity-20261020.json')})", '',
              '## Data and regeneration', '',
              'The [normalized data](data/model-performance.json) record source revisions, SHA-256 hashes, unrounded throughputs and per-point ratios. Gain is `100 × (selected / baseline − 1)`.', '']
    if native:
        lines += [f"These figures are generated by [vLLM Mach's renderer](https://github.com/troycheng/vllm-mach/blob/main/tools/render_performance_docs.py). The native-only kernel and earlier service studies remain in [benchmarks](benchmarks.md).", '']
    else:
        lines += ['```bash', 'python -m pip install -r docs/data/requirements-plot.txt', 'python tools/render_performance_docs.py --historical', 'python tools/render_performance_docs.py --historical --check', '```', '',
                  'Current report generation accepts `--mirror /path/to/mxfp6_sm120`. The historical report remains in Mach; original receipt JSON and per-study reports retain the source evidence.', '']
    return '\n'.join(lines)


def chart_names(data):
    names = [asset_name(data, m, 'throughput') for m in data['models']]
    names += [asset_name(data, m, 'precision') for m in data['models'] if 'fidelity_key' in m or m.get('mxfp8') or m.get('precision_comparison')]
    return names


def plots(data, dest, preview_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none',
                         'svg.hashsalt':'mach-performance-20261012', 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold',
                         'text.color':CHART_THEME['text'][0], 'axes.labelcolor':CHART_THEME['text'][0],
                         'xtick.color':CHART_THEME['text'][0], 'ytick.color':CHART_THEME['text'][0],
                         'axes.edgecolor':CHART_THEME['axis'][0], 'grid.color':CHART_THEME['grid'][0],
                         'hatch.linewidth':.6})
    colors = {key:CHART_THEME[key][0] for key in ('fp8','nvfp4','default','full')}
    dest.mkdir(parents=True,exist_ok=True)

    def save(fig, name):
        fig.savefig(dest/name,metadata={'Date':None},facecolor='white')
        if preview_dir:
            preview_dir.mkdir(parents=True,exist_ok=True)
            fig.savefig(preview_dir/Path(name).with_suffix('.png'),dpi=150,facecolor='white')
        plt.close(fig)
        path=dest/name
        path.write_text('\n'.join(line.rstrip() for line in themed_svg(path.read_text()).splitlines())+'\n')

    for m in data['models']:
        large='nvfp4_tps' in m['points'][0]
        block=m['id']=='qwen35-4b-block-fp8'
        series=throughput_series(m)
        fig,ax=plt.subplots(figsize=(11.6,4.6))
        x=list(range(len(m['points']))); width=.8/len(series)
        ymax=max(p[field] for p in m['points'] for _,field,_,_ in series)
        for j,(label,field,color,hatch) in enumerate(series):
            positions=[v+(j-(len(series)-1)/2)*width for v in x]
            bars=ax.bar(positions,[p[field] for p in m['points']],width*.9,
                        color=colors[color],label=label,hatch=hatch,
                        edgecolor='#ffffff',linewidth=.5,zorder=3)
            ax.bar_label(bars,labels=[f"{p[field]:,.0f}" for p in m['points']],padding=4,fontsize=10)
        for i,p in enumerate(m['points']):
            top=max(p[field] for _,field,_,_ in series)
            gain=f"{p['gain_pct']:+.1f}% vs {m.get('gain_reference_label', 'FP8')}"
            if m.get('mxfp8'):
                gain=f"{m.get('selected_label', 'FP32')} {p['gain_pct']:+.1f}%"
            if large:
                nvfp4_gain = 100 * (p['selected_tps'] / p['nvfp4_tps'] - 1)
                gain=f"{m.get('selected_label', 'Full')} {p['gain_pct']:+.1f}% vs FP8\n{nvfp4_gain:+.1f}% vs NVFP4"
            ax.text(i,top+ymax*.075,gain,ha='center',va='bottom',fontsize=10,
                    color=CHART_THEME['emphasis'][0],fontweight='bold',linespacing=1.4)
        ax.set(xlabel='Concurrent requests',ylabel='Output tokens/s',
               xticks=x,xticklabels=[f"c{p['c']}" for p in m['points']],
               ylim=(0,ymax*(1.32 if large else 1.24)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:,.0f}'))
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.65)
        ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.005),
                  ncols=len(series),borderaxespad=0)
        fig.suptitle(f"{m['title']} — throughput",fontsize=18,fontweight='bold',x=.09,ha='left',y=.975)
        caption = f'{runtime_caption(data)} · ' if data.get('runtime') else 'Mach: '
        fig.text(.09,.89,f"{caption}{m['tp']} × RTX 5090 · TP{m['tp']} · 3,000 input / 1,000 output tokens",fontsize=10,color=CHART_THEME['muted'][0])
        fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.78)
        save(fig,asset_name(data,m,'throughput'))
        if m.get('precision_comparison'):
            quality = data['fidelity'][m['precision_comparison']]
            fig, ax = plt.subplots(figsize=(11.6, 3.7))
            for j, (field, label, color) in enumerate([
                    ('reference_mae', quality['reference_label'], 'fp8'),
                    ('mach_mae', quality['candidate_label'], 'full')]):
                values = [p[field] for p in quality['points']]
                positions = [i + (j - .5) * .34 for i in range(len(values))]
                bars = ax.bar(positions, values, .3, color=colors[color], label=label, zorder=3)
                ax.bar_label(bars, labels=[f'{value:.5f}' for value in values], padding=4, fontsize=10)
            ymax = max(p[field] for p in quality['points'] for field in ('reference_mae', 'mach_mae'))
            ax.set(xticks=list(range(len(quality['points']))), xticklabels=[f"M{p['M']}" for p in quality['points']],
                   ylim=(0, ymax * 1.25), ylabel='Gold-token logprob MAE vs BF16', xlabel='Physical decode rows · lower error is better')
            ax.set_axisbelow(True); ax.grid(axis='y', alpha=.65)
            ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.005), ncols=2, borderaxespad=0)
            fig.suptitle(f"{m['title']} — precision", fontsize=18, fontweight='bold', x=.09, ha='left', y=.97)
            fig.text(.09, .865, f"{runtime_caption(data)} · {quality['queries']} queries / {quality['gold_tokens']:,} fixed tokens",
                     fontsize=10, color=CHART_THEME['muted'][0])
            fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.74)
            save(fig, asset_name(data, m, 'precision'))
        if m.get('mxfp8'):
            quality=data['fidelity']['mxfp8'][m['id'].removeprefix('qwen35-')]
            fig,ax=plt.subplots(figsize=(11.6,3.7))
            for j,(dtype,label,color) in enumerate([
                    ('float32','FP32 SSM','full'), ('bfloat16','Optional BF16 SSM','default')]):
                values=[p[dtype]['mae'] for p in quality['points']]
                bounds=[[p[dtype]['mae']-p[dtype]['ci95'][0] for p in quality['points']],
                        [p[dtype]['ci95'][1]-p[dtype]['mae'] for p in quality['points']]]
                positions=[i+(j-.5)*.34 for i in range(len(values))]
                bars=ax.bar(positions,values,.3,color=colors[color],label=label,
                            yerr=bounds,error_kw={'ecolor':CHART_THEME['interval'][0],'capsize':4,'elinewidth':1},zorder=3)
                for i,p in enumerate(quality['points']):
                    ax.text(positions[i],p[dtype]['ci95'][1]+.001,
                            f"{p[dtype]['mae']:.5f}",ha='center',fontsize=10)
            ymax=max(p[dtype]['ci95'][1] for p in quality['points'] for dtype in ('float32','bfloat16'))
            ax.set(xticks=[0,1],xticklabels=['M32','M64'],ylim=(0,ymax*1.2),
                   ylabel='Gold-token logprob MAE vs BF16',xlabel='Physical decode rows · lower error is better')
            ax.set_axisbelow(True);ax.grid(axis='y',alpha=.65)
            ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.005),ncols=2,borderaxespad=0)
            fig.suptitle(f"{m['title']} — precision",fontsize=18,fontweight='bold',x=.09,ha='left',y=.97)
            fig.text(.09,.865,(runtime_caption(data)+' · ' if data.get('asset_suffix') else 'October 20 · ')+'256 queries / 10,479 fixed tokens · error bars: 95% query-bootstrap intervals',
                     fontsize=10,color=CHART_THEME['muted'][0])
            fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.74)
            save(fig,asset_name(data,m,'precision'))
        if 'fidelity_key' not in m:
            continue
        quality=data['fidelity']['native'][m['fidelity_key']]
        intervals=data['fidelity']['native_ci95'][m['fidelity_key']]
        keys=['fp8','nvfp4','default','full']
        labels=['FP8', 'NVFP4 (ScaleSweep)' if m['fidelity_key']=='moe' else 'NVFP4',
                'MXFP6 default','MXFP6 full']
        labels=[m.get('precision_labels', {}).get(k, label) for k,label in zip(keys,labels)]
        values=[quality[k] for k in keys]
        bounds=[[quality[k]-intervals[k][0] for k in keys],
                [intervals[k][1]-quality[k] for k in keys]]
        fig,ax=plt.subplots(figsize=(11.6,3.5))
        bars=ax.barh(list(range(4)),values,height=.55,color=[colors[k] for k in keys],
                     xerr=bounds,error_kw={'ecolor':CHART_THEME['interval'][0],'capsize':3,'elinewidth':1},zorder=3)
        bars[1].set_hatch('//');bars[1].set_edgecolor('#ffffff')
        for i,k in enumerate(keys):
            ax.text(intervals[k][1]+.004,i,f'{quality[k]:.5f}',va='center',fontsize=11)
        ax.set(yticks=list(range(4)),yticklabels=labels,xlim=(0,.285),
               xlabel='Gold-token logprob MAE vs BF16 · lower is better')
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.65)
        fig.suptitle(f"{m['title']} — precision",fontsize=18,fontweight='bold',x=.09,ha='left',y=.97)
        fig.text(.09,.86,(runtime_caption(data)+' · ' if data.get('asset_suffix') else '')+'M32 · 256 queries / 10,479 fixed tokens · error bars: 95% query-bootstrap intervals',fontsize=10,color=CHART_THEME['muted'][0])
        fig.subplots_adjust(left=.23,right=.97,bottom=.19,top=.78)
        save(fig,asset_name(data,m,'precision'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--mirror',type=Path)
    parser.add_argument('--preview-dir',type=Path,help='Optional local PNG previews; published charts use SVG.')
    parser.add_argument('--migration-data',type=Path,help='Finalized accepted normalized dataset; separate runtime-suffixed outputs.')
    parser.add_argument('--historical', action='store_true', help='Regenerate the preserved vLLM0.29 report explicitly.')
    args=parser.parse_args()
    if args.historical and args.migration_data:
        parser.error('--historical and --migration-data are mutually exclusive')
    current = not args.historical
    published_data = DATA / 'model-performance-vllm033.json'
    selected_data = args.migration_data or published_data
    data=migration_data(selected_data) if current else dataset()
    release_table = vllm034_table() if current and selected_data.resolve() == published_data else None
    output_suffix=data.get('asset_suffix','')
    with TemporaryDirectory() as tmp:
        rendered=Path(tmp)
        plots(data,rendered,args.preview_dir)
        for root,native in [(ROOT,False)]+([(args.mirror,True)] if args.mirror and current else []):
            exported = dict(data)
            targets={root/f'docs/data/model-performance{output_suffix}.json':json.dumps(exported,indent=2)+'\n',
                     root/('docs/performance.md' if current else 'docs/performance-vllm029.md'):migration_markdown(data,native,release_table) if current else markdown(data,native)}
            if current:
                targets[root/'README.md'] = update_readme((root/'README.md').read_text(), readme_performance(data, native, release_table))
            targets.update({root/'docs/assets'/name:(rendered/name).read_text() for name in chart_names(data)})
            if args.check:
                for path,text in targets.items():assert path.read_text()==text,path
            else:
                for path,text in targets.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
                for stem in (() if current else ('model-throughput','model-gains')):
                    for suffix in ('.svg','.png'):
                        (root/'docs/assets'/f'{stem}{suffix}').unlink(missing_ok=True)
    print(migration_markdown(data, release_table=release_table) if current else overview(data))


if __name__=='__main__':
    main()
