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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy-only', action='store_true')
    args = parser.parse_args()
    accuracy = json.loads((HERE/'accuracy-comparison-m32-20260910.json').read_text())
    assert set(ORDER) == set(accuracy['runs'])
    fig, ax = plt.subplots(figsize=(12.8,5.4))
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
    ax.set_xlabel('Mean absolute error (lower is better)',labelpad=13)
    frame(ax)
    save(fig,'accuracy-comparison')

    if args.accuracy_only:
        print('Rendered six-arm accuracy comparison (PNG + SVG)')
        return
    performance = json.loads((HERE/'quantization-comparison-3k1k-20260910.json').read_text())
    assert set(ORDER) == set(performance['runs'])

    fig, ax = plt.subplots(figsize=(12.8,6.0))
    fig.subplots_adjust(left=.09,right=.97,top=.79,bottom=.15)
    for name in ORDER:
        points = performance['runs'][name]['points']
        assert [p['concurrency'] for p in points] == [4,16,24,32]
        ax.plot([p['concurrency'] for p in points],
                [p['output_throughput_tokens_per_s'] for p in points],
                label=LABELS[name],color=COLORS[name],marker=MARKERS[name],
                linestyle=STYLES[name],lw=2.0,ms=7,
                markerfacecolor='white' if name in ('fp8_stock029','k4k5_derived_w6_fp16') else COLORS[name])
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
    save(fig,'throughput-comparison')
    plot_tradeoff(accuracy,performance)
    print('Rendered three title-free comparison figures (PNG + SVG)')

if __name__ == '__main__':
    main()
