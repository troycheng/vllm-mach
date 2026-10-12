#!/usr/bin/env python3
"""Render model-level serving tables and plots from checked-in measurements."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data'
SOURCE_REV = 'eaa0acd6f5700913d10a422f6d4075614b9fb6be'
SOURCE_BASE = f'https://github.com/troycheng/vllm-mach/blob/{SOURCE_REV}/docs/'
GUIDE_BASE = 'https://github.com/troycheng/vllm-mach/blob/main/docs/'
SOURCES = {}


def load(name):
    path = DATA / name
    SOURCES[name] = dict(url=SOURCE_BASE + 'data/' + name,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return json.loads(path.read_text())


def point(c, baseline, selected, **extra):
    assert baseline > 0 and selected > 0
    return dict(c=int(c), baseline_tps=baseline, selected_tps=selected,
                gain_pct=100 * (selected / baseline - 1), **extra)


def dataset():
    two = load('mxfp8-2b-triton-hybrid-sixpoint-20261011.json')
    four = load('mxfp8-qualification-20261006.json')
    block = load('fp8-qualification-20261006.json')
    dense = load('dense-gains-aligned-baseline-20260920.json')
    core = load('dense-core-ablation-20260920.json')
    moe = load('moe-recheck-20260928.json')
    dense_nvfp4 = load('native-serving.json')
    moe_nvfp4 = load('qwen35-default-full-20260917.json')
    fidelity = load('profile-fidelity-20260917.json')
    models = [
        dict(id='qwen35-2b', title='Qwen3.5-2B', profile='MXFP8 · Triton + hybrid', tp=1,
             date='2026-10-11', guide='mxfp8-2b-champion.md',
             scope='Selected and previous configurations are paired on GPU0; community FP8 is the historical GPU1 reference.',
             source='mxfp8-2b-triton-hybrid-sixpoint-20261011.json',
             points=[point(x['c'], x['community_fp8_tps'], x['new']['output_tps'],
                           previous_tps=x['previous']['output_tps'],
                           previous_gain_pct=100*(x['new']['output_tps']/x['previous']['output_tps']-1),
                           mean_ttft_ms=x['new']['mean_ttft_ms'], mean_tpot_ms=x['new']['mean_tpot_ms'])
                     for x in two['throughput']['points']]),
        dict(id='qwen35-4b', title='Qwen3.5-4B', profile='MXFP8 champion', tp=1,
             date='2026-10-06', guide='mxfp8-champion.md',
             scope='One complete source-built champion screen versus the October 5 community FP8 screen; 1,040 requests per screen.',
             source='mxfp8-qualification-20261006.json',
             points=[point(x['concurrency'], x['historical_community_fp8']['output_throughput_tokens_per_s'],
                           x['packaged']['output_throughput_tokens_per_s'],
                           mean_ttft_ms=x['packaged']['mean_ttft_ms'], mean_tpot_ms=x['packaged']['mean_tpot_ms'])
                     for x in four['sixpoint']['points']]),
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
    return dict(schema_version=2, updated='2026-10-12', unit='output tokens/s',
                workload=dict(input_tokens=3000, output_tokens=1000), gpu='RTX 5090 32 GiB',
                full_profile='MXFP6 full adds FP16 recurrent state and NVFP4 candidate search with BF16 head refinement.',
                sources=SOURCES, models=models,
                fidelity=dict(date_native='2026-09-17',
                    two_b={str(p['rows']):p['new_mae'] for p in two['precision']['points']},
                    four_b={str(p['physical_rows']):p['comparison']['MAE_raw_gold_logprob_vs_same_M_BF16'] for p in four['quality']},
                    native={name:{arm:run['mae'] for arm,run in m['m32']['runs'].items()}
                            for name,m in fidelity['models'].items()},
                    native_ci95={name:{arm:run['ci95'] for arm,run in m['m32']['runs'].items()}
                                 for name,m in fidelity['models'].items()},
                    bootstrap=fidelity['bootstrap']))


def overview(data):
    lines = ['| Model / profile | GPUs | Throughput gain vs FP8 | Details |',
             '|---|---|---:|---|']
    for m in data['models']:
        gains = [p['gain_pct'] for p in m['points']]
        label = f"{min(gains):+.2f}% to {max(gains):+.2f}%"
        if 'default_tps' in m['points'][0]:
            dg = [p['default_gain_pct'] for p in m['points']]
            label = f"default {min(dg):+.2f}% to {max(dg):+.2f}%; full {label}"
        lines.append(f"| {m['title']} · {m['profile']} | {m['tp']} · TP{m['tp']} | {label} | [Data](docs/performance.md#{m['id']}) |")
    return '\n'.join(lines)


def markdown(data, native=False):
    def guide(path):
        if native:
            return (SOURCE_BASE if path.startswith('data/') else GUIDE_BASE) + path
        return path
    q=data['fidelity']; dense_q=q['native']['dense']; moe_q=q['native']['moe']
    lines = ['# Model serving performance', '',
             'Measured complete configurations on RTX 5090: **3,000 input / 1,000 output tokens**, text-only, non-speculative inference. Throughput is **output tokens/s**; `c` is concurrent requests, and physical kernel rows are denoted `M`.', '',
             'Each model has its own throughput chart. The 27B Dense and 35B-A3B MoE comparisons include FP8, NVFP4 and both MXFP6 profiles, followed by a separate precision chart against BF16.', '',
             'MXFP6 **default** retains FP32 recurrent state and the BF16 head. **Full** adds FP16 state and NVFP4 candidate search with BF16 refinement. The 2B MXFP8 champion uses BF16 KV and K128 hybrid head at M≤32; 4B uses calibrated FP8 KV and its own refined head.', '']
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
        elif m['id']=='qwen35-2b':
            lines += ['| Concurrency | Community FP8 | Previous FA2/full-head | Current Triton + hybrid | vs FP8 | vs previous |', '|---|---:|---:|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['baseline_tps']:.2f} | {p['previous_tps']:.2f} | {p['selected_tps']:.2f} | {p['gain_pct']:+.2f}% | {p['previous_gain_pct']:+.2f}% |")
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
              'Precision is measured as token-logprob MAE against BF16 using 256 queries / 10,479 tokens per physical-row setting. Lower MAE means closer agreement with BF16.', '',
              '| Profile | Physical rows | Numerical result |', '|---|---|---|',
              f"| 2B MXFP8 Triton + hybrid | M4/16/24/32/48/64 | BF16-anchor MAE {min(q['two_b'].values()):.5f}–{max(q['two_b'].values()):.5f}; M32 +2.58% versus FA2. Actual-generation head checks: 129,878 decisions, no omitted BF16 winner or lower-scored selection, one equal-score tie. |",
              f"| 4B MXFP8 | M32/64 | Same-M BF16 MAE {q['four_b']['32']:.5f} / {q['four_b']['64']:.5f}. |",
              '| 4B block-FP8 | M4/32/64 | Fixed-gold records byte-identical to stock block-FP8. |',
              f"| 27B Dense MXFP6 default / full | M32 | BF16 MAE {dense_q['default']:.5f} / {dense_q['full']:.5f}; FP8 {dense_q['fp8']:.5f}, NVFP4 {dense_q['nvfp4']:.5f}. |",
              f"| 35B-A3B MoE MXFP6 default / full | M32 | BF16 MAE {moe_q['default']:.5f} / {moe_q['full']:.5f}; FP8 {moe_q['fp8']:.5f}, NVFP4 {moe_q['nvfp4']:.5f}. |", '',
              f"[2B precision and generation coverage]({guide('mxfp8-2b-champion.md#precision')}) · [4B qualification]({guide('mxfp8-champion.md#validation')}) · [MXFP6 fidelity methodology]({guide('profile-fidelity-20260917.md')})", '',
              '### Measurement protocol', '',
              'The linked studies and [source records](data/model-performance.json) retain hardware identities, software revisions, request counts and baseline dates. MXFP6 precision results use the September 17 evaluation. MXFP6 and 2B logprob checks use the complete BF16 head; hybrid-head validation uses actual generation. The fixed-gold corpus is private.', '',
              'The 27B NVFP4 throughput reference is the September 15 token-ID sweep. The 35B-A3B NVFP4 throughput reference is the September 17 user service; its launch configuration was not recorded. The separate 35B-A3B precision evaluation used the NVFP4 ScaleSweep checkpoint. Source records preserve these identities and measurement dates.', '',
              '## Data and regeneration', '',
              'The [normalized data](data/model-performance.json) record source revisions, SHA-256 hashes, unrounded throughputs and per-point ratios. Gain is `100 × (selected / baseline − 1)`.', '']
    if native:
        lines += [f"These figures are generated by [vLLM Mach's renderer](https://github.com/troycheng/vllm-mach/blob/main/tools/render_performance_docs.py). The native-only kernel and earlier service studies remain in [benchmarks](benchmarks.md).", '']
    else:
        lines += ['```bash', 'python -m pip install -r docs/data/requirements-plot.txt', 'python tools/render_performance_docs.py', 'python tools/render_performance_docs.py --check', '```', '',
                  'Use `--mirror /path/to/mxfp6_sm120` when updating the shared service overview in both projects. Original receipt JSON and per-study reports remain the source evidence.', '']
    return '\n'.join(lines)


def chart_names(data):
    names = [f"{m['id']}-throughput.svg" for m in data['models']]
    names += [f"{m['id']}-precision.svg" for m in data['models'] if 'fidelity_key' in m]
    return names


def plots(data, dest, preview_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none',
                         'svg.hashsalt':'mach-performance-20261012', 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold',
                         'text.color':'#202936', 'axes.labelcolor':'#202936',
                         'hatch.linewidth':.6})
    colors = {'fp8':'#8994a4', 'nvfp4':'#d58a21', 'default':'#537da2', 'full':'#087f8c'}
    dest.mkdir(parents=True,exist_ok=True)

    def save(fig, name):
        fig.savefig(dest/name,metadata={'Date':None},facecolor='white')
        if preview_dir:
            preview_dir.mkdir(parents=True,exist_ok=True)
            fig.savefig(preview_dir/Path(name).with_suffix('.png'),dpi=150,facecolor='white')
        plt.close(fig)
        path=dest/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

    for m in data['models']:
        large='nvfp4_tps' in m['points'][0]
        block=m['id']=='qwen35-4b-block-fp8'
        series=[('FP8 baseline', 'baseline_tps', 'fp8', '')]
        if large:
            series += [('NVFP4 reference', 'nvfp4_tps', 'nvfp4', '//'),
                       ('MXFP6 default', 'default_tps', 'default', ''),
                       ('MXFP6 full', 'selected_tps', 'full', '')]
        else:
            label='Mach block-FP8' if block else 'MXFP8 champion'
            series += [(label, 'selected_tps', 'full', '')]
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
            gain=f"{p['gain_pct']:+.1f}% vs FP8"
            if large:
                gain=f"Full {p['gain_pct']:+.1f}% vs FP8\n{p['nvfp4_gain_pct']:+.1f}% vs NVFP4"
            ax.text(i,top+ymax*.075,gain,ha='center',va='bottom',fontsize=10,
                    color=colors['full'],fontweight='bold',linespacing=1.4)
        ax.set(xlabel='Concurrent requests',ylabel='Output tokens/s',
               xticks=x,xticklabels=[f"c{p['c']}" for p in m['points']],
               ylim=(0,ymax*(1.32 if large else 1.24)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:,.0f}'))
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
        ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.005),
                  ncols=len(series),borderaxespad=0)
        fig.suptitle(f"{m['title']} — throughput",fontsize=18,fontweight='bold',x=.09,ha='left',y=.975)
        fig.text(.09,.89,f"Mach: {m['tp']} × RTX 5090 · TP{m['tp']} · 3,000 input / 1,000 output tokens",fontsize=10,color='#4b5563')
        fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.78)
        save(fig,f"{m['id']}-throughput.svg")
        if not large:
            continue
        quality=data['fidelity']['native'][m['fidelity_key']]
        intervals=data['fidelity']['native_ci95'][m['fidelity_key']]
        keys=['fp8','nvfp4','default','full']
        labels=['FP8', 'NVFP4 (ScaleSweep)' if m['fidelity_key']=='moe' else 'NVFP4',
                'MXFP6 default','MXFP6 full']
        values=[quality[k] for k in keys]
        bounds=[[quality[k]-intervals[k][0] for k in keys],
                [intervals[k][1]-quality[k] for k in keys]]
        fig,ax=plt.subplots(figsize=(11.6,3.5))
        bars=ax.barh(list(range(4)),values,height=.55,color=[colors[k] for k in keys],
                     xerr=bounds,error_kw={'ecolor':'#354152','capsize':3,'elinewidth':1},zorder=3)
        bars[1].set_hatch('//');bars[1].set_edgecolor('#ffffff')
        for i,k in enumerate(keys):
            ax.text(intervals[k][1]+.004,i,f'{quality[k]:.5f}',va='center',fontsize=11)
        ax.set(yticks=list(range(4)),yticklabels=labels,xlim=(0,.285),
               xlabel='Gold-token logprob MAE vs BF16 · lower is better')
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
        fig.suptitle(f"{m['title']} — precision",fontsize=18,fontweight='bold',x=.09,ha='left',y=.97)
        fig.text(.09,.86,'M32 · 256 queries / 10,479 fixed tokens · error bars: 95% query-bootstrap intervals',fontsize=10,color='#4b5563')
        fig.subplots_adjust(left=.23,right=.97,bottom=.19,top=.78)
        save(fig,f"{m['id']}-precision.svg")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--mirror',type=Path)
    parser.add_argument('--preview-dir',type=Path,help='Optional local PNG previews; published charts use SVG.')
    args=parser.parse_args();data=dataset()
    with TemporaryDirectory() as tmp:
        rendered=Path(tmp)
        plots(data,rendered,args.preview_dir)
        for root,native in [(ROOT,False)]+([(args.mirror,True)] if args.mirror else []):
            targets={root/'docs/data/model-performance.json':json.dumps(data,indent=2)+'\n',
                     root/'docs/performance.md':markdown(data,native)}
            targets.update({root/'docs/assets'/name:(rendered/name).read_text() for name in chart_names(data)})
            if args.check:
                for path,text in targets.items():assert path.read_text()==text,path
            else:
                for path,text in targets.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
                for stem in ('model-throughput','model-gains'):
                    for suffix in ('.svg','.png'):
                        (root/'docs/assets'/f'{stem}{suffix}').unlink(missing_ok=True)
    print(overview(data))


if __name__=='__main__':
    main()
