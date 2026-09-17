"""Render the README figures from checked data. Requires matplotlib and numpy."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/'images'
ORDER = ['fp8_stock029', 'fp8_accelerated028', 'mxfp6_champion',
         'k5k6_hybrid_fp16_ba', 'k4k5_derived_w6_fp16', 'nvfp4']
LABELS = {
    'fp8_stock029': 'FP8 · official vLLM 0.29',
    'fp8_accelerated028': 'FP8 · accelerated vLLM 0.28',
    'mxfp6_champion': 'MXFP6 Champion',
    'k5k6_hybrid_fp16_ba': 'K5/K6 Hybrid + FP16 SSM + BA',
    'k4k5_derived_w6_fp16': 'K4/K5-derived W6 + FP16 SSM',
    'nvfp4': 'NVFP4 · local calibration',
}
COLORS = dict(zip(ORDER, ['#87919D','#3B69C8','#343E4A','#B58A2B','#788547','#D77B44']))
MARKERS = dict(zip(ORDER, ['o','s','D','^','v','P']))
STYLES = dict(zip(ORDER, ['--','-','-.','-','--',':']))
INK = '#202A35'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13,
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':'#657180',
    'ytick.color':INK, 'axes.edgecolor':'#C5CBD1', 'axes.linewidth':.8,
    'svg.fonttype':'none', 'svg.hashsalt':'vllm-mach-comparison-20260910',
    'savefig.facecolor':'white', 'figure.facecolor':'white'})

def frame(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both', length=0, pad=9)
    ax.set_axisbelow(True)

def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(stem+'.png'), dpi=180)
    fig.savefig(OUT/(stem+'.svg'), metadata={'Date':None})
    svg = OUT/(stem+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

def tradeoff_points(accuracy, performance):
    baseline = performance['runs']['fp8_stock029']['points']
    assert [p['concurrency'] for p in baseline] == [4,16,24,32]
    result = {}
    for name in ORDER:
        points = performance['runs'][name]['points']
        assert [p['concurrency'] for p in points] == [4,16,24,32]
        gains = [100*(p['output_throughput_tokens_per_s']/b['output_throughput_tokens_per_s']-1)
                 for p,b in zip(points,baseline,strict=True)]
        result[name] = {'mae':accuracy['runs'][name]['mae'],
                        'ci95':accuracy['runs'][name]['ci95'],
                        'gains':gains, 'mean':float(np.mean(gains)),
                        'min':min(gains), 'max':max(gains)}
    assert result['fp8_stock029']['gains'] == [0,0,0,0]
    return result

def plot_tradeoff(accuracy, performance):
    points = tradeoff_points(accuracy,performance)
    fig, ax = plt.subplots(figsize=(12.8,7.0))
    fig.subplots_adjust(left=.09,right=.97,top=.97,bottom=.14)
    # Place labels around the measured coordinates, without jittering the data.
    positions = {
        'fp8_stock029': (.065,-4),
        'fp8_accelerated028': (.064,8.5),
        'mxfp6_champion': (.111,19),
        'k4k5_derived_w6_fp16': (.111,32.5),
        'k5k6_hybrid_fp16_ba': (.045,52),
        'nvfp4': (.151,56),
    }
    for name in ORDER:
        p = points[name]; x=p['mae']; y=p['mean']; lo,hi=p['ci95']
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],yerr=[[y-p['min']],[p['max']-y]],
                    fmt=MARKERS[name],color=COLORS[name],ecolor=COLORS[name],
                    markersize=8,elinewidth=1.15,capsize=4,markeredgewidth=1.3,
                    markerfacecolor='white' if name in ('fp8_stock029','k4k5_derived_w6_fp16') else COLORS[name],
                    zorder=3)
        detail = f"{x:.5f} MAE · {y:+.1f}% avg" if name != 'fp8_stock029' else f'{x:.5f} MAE · baseline'
        ax.annotate(LABELS[name]+'\n'+detail,xy=(x,y),xytext=positions[name],
                    textcoords='data',fontsize=11.5,linespacing=1.5,ha='left',va='center',
                    color=INK,arrowprops={'arrowstyle':'-','color':COLORS[name],
                                        'lw':.9,'shrinkA':6,'shrinkB':7},
                    bbox={'facecolor':'white','edgecolor':'none','pad':2},zorder=4)
    ax.axhline(0,color='#AAB2BC',lw=1,zorder=1)
    ax.set_xlim(.035,.205); ax.set_ylim(-10,62)
    ax.xaxis.set_major_locator(MultipleLocator(.025))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
    ax.set_yticks([0,10,20,30,40,50,60])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos: '0%' if value==0 else f'{value:+.0f}%'))
    ax.grid(color='#E8EBEE',lw=.8)
    ax.set_xlabel('Gold-token logprob MAE vs BF16 (lower is better)',labelpad=13)
    ax.set_ylabel('Output-throughput gain vs FP8',labelpad=15)
    frame(ax)
    save(fig,'quality-throughput-tradeoff')


def plot_native_tradeoff(accuracy, performance):
    baseline = performance['runs']['fp8']['points']
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    fig.subplots_adjust(left=.09, right=.97, top=.82, bottom=.16)
    for name in ORDER:
        run = accuracy['runs'][name]
        gains = [100*(p['output_throughput_tokens_per_s']/b['output_throughput_tokens_per_s']-1)
                 for p,b in zip(performance['runs'][name]['points'], baseline, strict=True)]
        x, y = run['mae'], float(np.mean(gains))
        lo, hi = run['ci95']
        ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], yerr=[[y-min(gains)], [max(gains)-y]],
                    fmt=MARKERS[name], color=COLORS[name], capsize=4, markersize=8,
                    label=f"{LABELS[name]} ({y:+.1f}%)")
    ax.axhline(0, color='#AAB2BC', lw=1)
    ax.grid(color='#E8EBEE', lw=.8)
    ax.set_xlabel('Physical M32 gold-token logprob MAE vs BF16', labelpad=13)
    ax.set_ylabel('Output-throughput gain vs stock FP8', labelpad=15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:+.0f}%'))
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.06,.995),
               ncol=2, frameon=False, fontsize=11.5)
    frame(ax)
    save(fig, 'quality-throughput-tradeoff')

def plot_gdn_m4(data):
    fig, ax = plt.subplots(figsize=(12.8,6.2))
    fig.subplots_adjust(left=.40, right=.94, top=.83, bottom=.20)
    for y, name in enumerate(['default', 'gdn', 'full_ba', 'full_gdn']):
        run = data['runs'][name]
        x = run['mae']; lo, hi = run['ci95']
        ax.barh(y, x, height=.5, color=COLORS[name])
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='none',ecolor=INK,capsize=4)
        ax.text(hi+.003,y,f'{x:.5f}',va='center')
    ax.set_yticks([0,1,2,3], [LABELS[n] for n in ['default', 'gdn', 'full_ba', 'full_gdn']])
    ax.invert_yaxis()
    ax.set_xlim(0,max(r['ci95'][1] for r in data['runs'].values())+.025)
    ax.set_xlabel('Physical M4: gold-token logprob MAE vs BF16',labelpad=13)
    fig.suptitle("256 queries · 10,479 gold tokens · 95% query-bootstrap intervals", fontsize=12, y=.96)
    ax.grid(axis='x',color='#E8EBEE',lw=.8)
    frame(ax)
    save(fig,'gdn-m4-fidelity')


def plot_gdn_ablation(performance):
    fig, ax = plt.subplots(figsize=(11.8,5.2))
    fig.subplots_adjust(left=.10, right=.97, top=.72, bottom=.17)
    x = np.arange(4)
    for index, (arm, baseline, label) in enumerate([
        ('persistent', 'default', 'Persistent only / previous default'),
        ('full_ba', 'full', 'BA overlap / previous full (FP16)'),
        ('gdn', 'default', 'Combined default / previous default'),
        ('full_gdn', 'full_ba', 'FP16 persistent / full without persistent'),
    ]):
        values = [100*(p['output_throughput_tokens_per_s']/b['output_throughput_tokens_per_s']-1)
                  for p,b in zip(performance['runs'][arm]['points'], performance['runs'][baseline]['points'], strict=True)]
        bars = ax.bar(x+(index-1.5)*.21, values, width=.20, color=COLORS[arm], label=label)
        ax.bar_label(bars, labels=[f'{v:+.1f}%' for v in values], padding=4, fontsize=10)
    ax.axhline(0, color='#AAB2BC', lw=1)
    ax.set_xticks(x, ['c4','c16','c24','c32'])
    ax.set_xlabel('Concurrent requests', labelpad=12)
    ax.set_ylabel('Output-throughput change', labelpad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:+.0f}%'))
    ax.grid(axis='y', color='#E8EBEE', lw=.8)
    ax.set_ylim(-2, 28)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.08,.99), frameon=False, fontsize=11)
    frame(ax)
    save(fig, 'gdn-throughput-ablation')


def plot_throughput(performance, order, labels, colors, markers, styles, stem, footnote=None):
    fig, ax = plt.subplots(figsize=(12.8,6.6))
    fig.subplots_adjust(left=.09,right=.97,top=.82,bottom=.21 if footnote else .15)
    if footnote:
        fig.text(.09, .025, footnote, fontsize=10, color='#657180', linespacing=1.5)
    for name in order:
        points = performance['runs'][name]['points']
        assert [p['concurrency'] for p in points] == [4,16,24,32]
        ax.plot([p['concurrency'] for p in points],
                [p['output_throughput_tokens_per_s'] for p in points],
                label=labels[name],color=colors[name],marker=markers[name],
                linestyle=styles[name],lw=2.0,ms=7,
                markerfacecolor='white' if name in ('fp8_stock029','k4k5_derived_w6_fp16') else colors[name])
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.06,.995),
               ncol=2,frameon=False,fontsize=12,columnspacing=2.8,handlelength=2.8,labelspacing=.55)
    peak = max(p['output_throughput_tokens_per_s'] for run in performance['runs'].values() for p in run['points'])
    ax.set_xlim(2.8,33.2); ax.set_ylim(0,max(1800,200*np.ceil(peak*1.08/200)))
    ax.set_xticks([4,16,24,32],['c4','c16','c24','c32'])
    ax.yaxis.set_major_locator(MultipleLocator(400))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis='y',color='#E8EBEE',lw=.8)
    ax.set_ylabel('Output tokens/s',labelpad=15)
    ax.set_xlabel('Concurrent requests',labelpad=12)
    frame(ax)
    save(fig, stem)


def plot_moe_throughput():
    optimized = json.loads((HERE/'qwen35-default-full-20260917.json').read_text())
    order = ['fp8', 'nvfp4', 'default', 'full']
    runs = {}
    for name in order:
        source = optimized
        runs[name] = {'points': [
            {'concurrency': row['concurrency'],
             'output_throughput_tokens_per_s': row[name]['aggregate']['output_throughput_tokens_per_s']}
            for row in source['comparisons'] if row['concurrency'] in (4, 16, 24, 32)
        ]}
    plot_throughput(
        {'runs': runs}, order,
        {'fp8': 'FP8 · vLLM 0.29 baseline',
         'nvfp4': 'NVFP4 · vLLM 0.29 baseline',
         'default': 'MXFP6 · Mach default',
         'full': 'MXFP6 · Mach full'},
        {'fp8': '#87919D', 'nvfp4': '#D77B44', 'default': '#126149', 'full': '#8050A0'},
        {'fp8': 'o', 'nvfp4': 's', 'default': 'X', 'full': '*'},
        {'fp8': '--', 'nvfp4': ':', 'default': '-', 'full': '-'},
        'qwen35-moe-throughput',
        footnote='Qwen3.5-35B-A3B · 3000 input / 1000 output tokens · 2-run means · September 17, 2026\n'
                 'FP8 / NVFP4: user-provided vLLM baseline services, remeasured at c4 / c16 / c24 / c32.\n'
                 'Mach: 2 × RTX 5090 · TP2 · 2048 batched tokens · max sequences 64.')


def main():
    global ORDER, LABELS, COLORS, MARKERS, STYLES
    parser = argparse.ArgumentParser()
    parser.add_argument('--moe-only', action='store_true', help='Render the 35B MoE throughput chart only')
    parser.add_argument('--accuracy-only', action='store_true')
    parser.add_argument('--gdn-m4-only', action='store_true')
    parser.add_argument('--legacy', action='store_true', help='Render archived EXL3 measurements to an archive directory')
    args = parser.parse_args()
    if args.moe_only:
        if args.legacy or args.accuracy_only or args.gdn_m4_only:
            parser.error('--moe-only cannot be combined with other rendering modes')
        plot_moe_throughput()
        return
    if not args.legacy:
        ORDER = ['fp8', 'default', 'persistent', 'gdn', 'full', 'full_ba', 'full_gdn', 'nvfp4']
        LABELS = {'fp8':'FP8 · stock vLLM 0.29 (Sep 15)',
                  'default':'MXFP6 · previous default', 'persistent':'MXFP6 · persistent only',
                  'gdn':'MXFP6 · Mach default',
                  'full':'MXFP6 · previous full', 'full_ba':'MXFP6 · full without persistent',
                  'full_gdn':'MXFP6 · Mach full',
                  'nvfp4':'NVFP4 · stock vLLM 0.29 (Sep 15)'}
        COLORS = dict(zip(ORDER, ['#87919D', '#3B69C8', '#159A98', '#126149', '#B58A2B', '#9579A6', '#8050A0', '#D77B44']))
        MARKERS = dict(zip(ORDER, ['o','s','D','X','^','v','*','P']))
        STYLES = dict(zip(ORDER, ['--','--',':','-','--','--','-',':']))
        ORDER = ['fp8', 'gdn', 'full_gdn', 'nvfp4']
    else:
        global OUT
        OUT = OUT/'historical'
        OUT.mkdir(parents=True, exist_ok=True)
    if args.gdn_m4_only:
        if args.legacy:
            parser.error('--gdn-m4-only cannot use --legacy')
        plot_gdn_m4(json.loads((HERE/'gdn-m4-fidelity.json').read_text()))
        return
    accuracy = json.loads((HERE/('accuracy-comparison-m32-20260910.json' if args.legacy else 'native-fidelity.json')).read_text())
    assert set(ORDER).issubset(accuracy['runs'])
    fig, ax = plt.subplots(figsize=(13.6,5.6))
    fig.subplots_adjust(left=.35,right=.96,top=.96,bottom=.17)
    for y, name in enumerate(ORDER):
        run = accuracy['runs'][name]
        v = run['mae']; lo,hi = run['ci95']
        ax.barh(y,v,height=.53,color=COLORS[name],edgecolor=COLORS[name],linewidth=.8)
        ax.errorbar(v,y,xerr=np.array([[v-lo],[hi-v]]),fmt='none',ecolor=INK,capsize=3,lw=1.1)
        ax.text(hi+.005,y,f'{v:.5f}',va='center',fontsize=12)
    ax.set_yticks(range(len(ORDER)),[LABELS[n] for n in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0,max(r['ci95'][1] for r in accuracy['runs'].values())+.043)
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    ax.grid(axis='x',color='#E8EBEE',lw=.8)
    ax.set_xlabel('Physical M32: gold-token logprob MAE vs BF16 (lower is better)',labelpad=13)
    frame(ax)
    save(fig,'accuracy-comparison')

    if not args.legacy and (HERE/'gdn-m4-fidelity.json').exists():
        plot_gdn_m4(json.loads((HERE/'gdn-m4-fidelity.json').read_text()))

    if args.accuracy_only:
        print('Rendered accuracy comparison (PNG + SVG)')
        return
    performance = json.loads((HERE/('quantization-comparison-3k1k-20260910.json' if args.legacy else 'native-serving.json')).read_text())
    assert set(ORDER).issubset(performance['runs'])

    plot_throughput(performance, ORDER, LABELS, COLORS, MARKERS, STYLES, 'throughput-comparison')
    (plot_tradeoff if args.legacy else plot_native_tradeoff)(accuracy,performance)
    if not args.legacy:
        plot_gdn_ablation(performance)
        plot_moe_throughput()
    print('Rendered three title-free comparison figures (PNG + SVG)')

if __name__ == '__main__':
    main()
