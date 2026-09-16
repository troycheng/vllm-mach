"""Dedicated TP2 optimization figures: decode workload is not the serving comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"svg.fonttype":"none", "svg.hashsalt":"mach-tp2-optimization"})


def save(fig, output, name):
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f"{name}.png",dpi=180)
    path=output/f"{name}.svg"
    fig.savefig(path,metadata={"Date":None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'images')
    p.add_argument('--validation',type=Path,help='Optional empty-output size and reverse-order checks')
    a=p.parse_args()
    stages=[json.loads(path.read_text()) for path in a.data]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    curves=[]
    for stage in stages:
        if 'matched_control' in stage:
            curves.append(stage['matched_control'])
        curves.append(stage)
    for index,stage in enumerate(curves):
        color=plt.get_cmap('tab20')(index % 20)
        rows=stage['decode']['rows']
        axes[0].errorbar([r['rows'] for r in rows],[r['mean_output_tokens_per_s'] for r in rows],
            yerr=[r['stdev_output_tokens_per_s'] for r in rows],marker='o',capsize=4,label=stage['label'],color=color)
        axes[1].plot([r['rows'] for r in rows],[r['mean_request_itl_ms'] for r in rows],marker='o',label=stage['label'],color=color)
    axes[0].set_ylabel('Output tokens/s, mean ± trial SD')
    axes[1].set_ylabel('Mean request ITL (ms)')
    for ax in axes:
        ax.set_xlabel('Logical requests (physical size verified separately)')
        ax.set_xticks([1,4,16,32]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('TP2 / FP32 SSM / BF16 head · 2048 input tokens\nB1: 129 output; B4/16/32: 1025 output · five trials · includes prefill')
    save(fig,a.output,'tp2-optimization-throughput')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,m in zip(axes,(4,32)):
        for i,stage in enumerate(stages):
            row=stage['fidelity'][str(m)]
            ax.errorbar(i,row['mae'],yerr=np.array([[row['mae']-row['ci95'][0]],[row['ci95'][1]-row['mae']]]),fmt='o',capsize=5)
        ax.set_xticks(range(len(stages)),[s['label'] for s in stages],rotation=20,ha='right');ax.set_title(f'Physical M{m}');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Gold logprob MAE vs BF16 / 95% query CI')
    fig.suptitle('Fresh teacher-forced scoring · 256 queries / 10,479 target tokens')
    save(fig,a.output,'tp2-optimization-fidelity')
    plt.close(fig)
    if a.validation:
        checks=json.loads(a.validation.read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        primary=checks['decode_trials']+checks['regression']['rows']
        groups=[(primary,[1,2,4,8,16,24,32],('off','regression-off')),
                (checks['reverse_order']['runs'],[32],('reverse-off',))]
        for ax,(records,sizes,controls) in zip(axes,groups):
            for probe,label,offset in [(False,'Zero core',-.08),(True,'Empty core',.08)]:
                selected={r['rows']:r for r in records if (r['mode'] not in controls)==probe}
                baseline={r['rows']:r['mean_output_tokens_per_s'] for r in records if r['mode'] in controls}
                ax.errorbar(np.arange(len(sizes))+offset,
                    [selected[m]['mean_output_tokens_per_s']/baseline[m] for m in sizes],
                    yerr=[selected[m]['stdev_output_tokens_per_s']/baseline[m] for m in sizes],
                    fmt='o',capsize=4,label=label)
            ax.axhline(1,color='grey',linestyle=':',linewidth=1)
            ax.set(xticks=np.arange(len(sizes)),xticklabels=sizes,xlabel='Physical M',
                   ylabel='Throughput / zero-core mean (± normalized trial SD)')
            ax.set_xlim(-.5,len(sizes)-.5)
            ax.grid(axis='y',alpha=.2);ax.legend()
        axes[0].set_title('GPUs 4/5 · five trials per arm')
        axes[1].set_title('GPUs 6/7 · empty first, zero second · five trials')
        fig.suptitle('GDN output initialization · independent matched experiments')
        save(fig,a.output,'tp2-empty-output-validation')
        plt.close(fig)


if __name__=='__main__':
    main()
