#!/usr/bin/env python3
"""Render model-level serving tables and plots from checked-in measurements."""
import argparse
import hashlib
import json
from pathlib import Path

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
    models.append(dict(id='qwen38-27b', title='Qwen3.8-27B Dense', profile='MXFP6 default / full', tp=2,
                       date='2026-09-20', guide='dense-gains-aligned-baseline-20260920.md',
                       scope='One configuration sweep; runtime, GPU-group and measurement-batch differences are recorded in the study.',
                       source='dense-gains-aligned-baseline-20260920.json',
                       points=[point(c, fp8[c]['output_throughput_tokens_per_s'], rows['dense_full']['throughput'][c],
                                     default_tps=rows['dense_default']['throughput'][c],
                                     default_gain_pct=100*(rows['dense_default']['throughput'][c]/fp8[c]['output_throughput_tokens_per_s']-1))
                               for c in ('4','16','24','32')]))
    models.append(dict(id='qwen35-35b-a3b', title='Qwen3.5-35B-A3B MoE', profile='MXFP6 default / full', tp=2,
                       date='2026-09-28', guide='moe-gains-fp8-baseline-20260920.md',
                       scope='MXFP6 means of two sweeps versus September 17 FP8. The measured default has projection tuning off; full includes it. Current launchers enable eligible tuning.',
                       source='moe-recheck-20260928.json',
                       points=[point(c, moe['baseline'][c]['aggregate']['output_throughput_tokens_per_s'],
                                     moe['arms']['projection']['points'][c]['throughput'],
                                     default_tps=moe['arms']['default']['points'][c]['throughput'],
                                     default_gain_pct=100*(moe['arms']['default']['points'][c]['throughput']/moe['baseline'][c]['aggregate']['output_throughput_tokens_per_s']-1))
                               for c in ('4','16','24','32')]))
    for model in models:
        assert [p['c'] for p in model['points']] == sorted({p['c'] for p in model['points']})
    return dict(schema_version=1, updated='2026-10-11', unit='output tokens/s',
                workload=dict(input_tokens=3000, output_tokens=1000), gpu='RTX 5090 32 GiB',
                full_profile='MXFP6 full adds FP16 recurrent state and NVFP4 candidate search with BF16 head refinement.',
                sources=SOURCES, models=models,
                fidelity=dict(date_native='2026-09-17',
                    two_b={str(p['rows']):p['new_mae'] for p in two['precision']['points']},
                    four_b={str(p['physical_rows']):p['comparison']['MAE_raw_gold_logprob_vs_same_M_BF16'] for p in four['quality']},
                    native={name:{arm:run['mae'] for arm,run in m['m32']['runs'].items()}
                            for name,m in fidelity['models'].items()}))


def overview(data):
    lines = ['| Model / profile | GPUs | Measured gain over its FP8 baseline | Details |',
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
             'The 2B/4B MXFP8 and 27B/35B MXFP6 paths combine native operators with vLLM Mach. Block-FP8 is a separate Mach path. Each model keeps its own measured FP8 baseline; the tables below state the comparison date and scope.', '',
             '![Serving throughput across four model sizes](assets/model-throughput.svg)', '',
             '![Per-point gain over each study\'s FP8 baseline](assets/model-gains.svg)', '',
             'MXFP6 **default** retains FP32 recurrent state and the BF16 head. **Full** adds FP16 state and NVFP4 candidate search with BF16 refinement. The 2B MXFP8 champion uses BF16 KV and K128 hybrid head at M≤32; 4B uses calibrated FP8 KV and its own refined head.', '']
    for m in data['models']:
        lines += [f"<a id=\"{m['id']}\"></a>", f"## {m['title']} — {m['profile']}", '',
                  f"{m['date']} · {m['tp']} × RTX 5090 · TP{m['tp']}. {m['scope']}", '']
        if 'default_tps' in m['points'][0]:
            lines += ['| Concurrency | FP8 | MXFP6 default | Gain | MXFP6 full | Gain |', '|---|---:|---:|---:|---:|---:|']
            for p in m['points']:
                lines.append(f"| c{p['c']} | {p['baseline_tps']:.2f} | {p['default_tps']:.2f} | {p['default_gain_pct']:+.2f}% | {p['selected_tps']:.2f} | {p['gain_pct']:+.2f}% |")
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
        if m['id']=='qwen35-2b':
            lines += ['A later same-runtime c32 observation was 7,216.58 tok/s, paired with FA2 + hybrid at 7,210.06 tok/s. It remains a separate comparison.', '']
    lines += ['## Numerical fidelity', '',
              'Fixed-gold checks use 256 queries / 10,479 token logprobs per tested physical-row setting. MAE is deviation from the named reference; it is not a task-accuracy score. The evaluation corpus is private. MXFP6 rows below come from the September 17 diagnostics. MXFP6 and 2B logprob requests use the complete BF16 head; the 2B head was also checked during actual generation.', '',
              '| Profile | Physical rows | Numerical result |', '|---|---|---|',
              f"| 2B MXFP8 Triton + hybrid | M4/16/24/32/48/64 | BF16-anchor MAE {min(q['two_b'].values()):.5f}–{max(q['two_b'].values()):.5f}; M32 +2.58% versus FA2. Actual-generation head checks: 129,878 decisions, no omitted BF16 winner or lower-scored selection, one equal-score tie. |",
              f"| 4B MXFP8 | M32/64 | Same-M BF16 MAE {q['four_b']['32']:.5f} / {q['four_b']['64']:.5f}; fixed-gold records unchanged from the accepted champion. |",
              '| 4B block-FP8 | M4/32/64 | Fixed-gold records byte-identical to stock block-FP8. |',
              f"| 27B Dense MXFP6 default / full | M32 | BF16 MAE {dense_q['default']:.5f} / {dense_q['full']:.5f}; FP8 {dense_q['fp8']:.5f}, NVFP4 {dense_q['nvfp4']:.5f}. |",
              f"| 35B-A3B MoE MXFP6 default / full | M32 | BF16 MAE {moe_q['default']:.5f} / {moe_q['full']:.5f}; FP8 {moe_q['fp8']:.5f}, NVFP4 {moe_q['nvfp4']:.5f}. |", '',
              f"[2B precision and generation coverage]({guide('mxfp8-2b-champion.md#precision')}) · [4B qualification]({guide('mxfp8-qualification-20261006.md')}) · [MXFP6 fidelity methodology]({guide('profile-fidelity-20260917.md')})", '',
              '## Data and regeneration', '',
              'The [normalized data](data/model-performance.json) record the source revision, SHA-256 hashes, unrounded throughputs and per-point ratios. Gain is `100 × (selected / baseline − 1)`. No gains from different experiments are multiplied together.', '']
    if native:
        lines += [f"These figures are generated by [vLLM Mach's renderer](https://github.com/troycheng/vllm-mach/blob/main/tools/render_performance_docs.py). The native-only kernel and earlier service studies remain in [benchmarks](benchmarks.md).", '']
    else:
        lines += ['```bash', 'python -m pip install -r docs/data/requirements-plot.txt', 'python tools/render_performance_docs.py', 'python tools/render_performance_docs.py --check', '```', '',
                  'Use `--mirror /path/to/mxfp6_sm120` when updating the shared service overview in both projects. Original receipt JSON and per-study reports remain the source evidence.', '']
    return '\n'.join(lines)


def plots(data, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none',
                         'svg.hashsalt':'mach-performance-20261011', 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold'})
    colors = {'baseline':'#677489','selected':'#087f8c','default':'#d58a21'}
    models=[m for m in data['models'] if m['id']!='qwen35-4b-block-fp8']
    fig,axes=plt.subplots(2,2,figsize=(12,8.0))
    for ax,m in zip(axes.flat,models):
        x=[p['c'] for p in m['points']]
        ax.plot(x,[p['baseline_tps'] for p in m['points']],color=colors['baseline'],marker='o',label='FP8 baseline',linewidth=2)
        if 'default_tps' in m['points'][0]:
            ax.plot(x,[p['default_tps'] for p in m['points']],color=colors['default'],marker='s',label='MXFP6 default',linewidth=2)
        label='MXFP6 full' if m['tp']==2 else 'MXFP8 champion'
        ax.plot(x,[p['selected_tps'] for p in m['points']],color=colors['selected'],marker='o',label=label,linewidth=2.3)
        ax.set_title(f"{m['title']} · TP{m['tp']}",loc='left',pad=12)
        ax.set(xlabel='Concurrent requests',ylabel='Output tokens/s',xticks=x,ylim=(0,None))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:,.0f}'))
        ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Serving throughput by model size',fontsize=19,fontweight='bold',x=.07,ha='left')
    fig.text(.07,.92,'RTX 5090 · 3,000 input / 1,000 output tokens · per-model baselines',fontsize=11,color='#4b5563')
    fig.text(.07,.02,'2B / 4B: TP1 MXFP8. 27B / 35B: TP2 MXFP6. Study dates and precision settings: docs/performance.md.',fontsize=9,color='#4b5563')
    fig.subplots_adjust(top=.85,bottom=.11,left=.085,right=.97,hspace=.42,wspace=.23)
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'model-throughput.svg',metadata={'Date':None},facecolor='white')
    fig.savefig(dest/'model-throughput.png',dpi=150,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.7))
    palette=['#087f8c','#7957b5','#346ea3','#d58a21','#ba4d51']
    for color,m in zip(palette,data['models']):
        label=m['title']+(' · full' if m['tp']==2 else ' · MXFP8' if 'block' not in m['id'] else '')
        ax.plot([p['c'] for p in m['points']],[p['gain_pct'] for p in m['points']],marker='o',linewidth=2,color=color,label=label)
    ax.set(xlabel='Concurrent requests',ylabel='Gain over own FP8 baseline (%)',xticks=[4,16,24,32,48,64],ylim=(0,None))
    ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,fontsize=9,loc='center left',bbox_to_anchor=(1.01,.5))
    ax.set_title('Measured gain at each concurrency',loc='left',fontsize=16,pad=14)
    fig.text(.09,.015,'Complete profile comparisons; different models and studies retain their own baseline and precision settings.',fontsize=9,color='#4b5563')
    fig.subplots_adjust(left=.09,right=.68,bottom=.17,top=.86)
    fig.savefig(dest/'model-gains.svg',metadata={'Date':None},facecolor='white')
    fig.savefig(dest/'model-gains.png',dpi=150,facecolor='white');plt.close(fig)
    for name in ('model-throughput.svg','model-gains.svg'):
        path=dest/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--mirror',type=Path)
    args=parser.parse_args();data=dataset()
    for root,native in [(ROOT,False)]+([(args.mirror,True)] if args.mirror else []):
        targets={root/'docs/data/model-performance.json':json.dumps(data,indent=2)+'\n',
                 root/'docs/performance.md':markdown(data,native)}
        if args.check:
            for path,text in targets.items():assert path.read_text()==text,path
            for name in ['model-throughput.svg','model-gains.svg']:
                assert (root/'docs/assets'/name).is_file(),name
        else:
            for path,text in targets.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            plots(data,root/'docs/assets')
    print(overview(data))


if __name__=='__main__':
    main()
