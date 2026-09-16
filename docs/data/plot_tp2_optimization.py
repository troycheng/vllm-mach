"""Dedicated P0/P1 figures: decode workload is not the serving comparison."""
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
    a=p.parse_args()
    stages=[json.loads(path.read_text()) for path in a.data]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for stage in stages:
        rows=stage['decode']['rows']
        axes[0].errorbar([r['rows'] for r in rows],[r['mean_output_tokens_per_s'] for r in rows],
            yerr=[r['stdev_output_tokens_per_s'] for r in rows],marker='o',capsize=4,label=stage['label'])
        axes[1].plot([r['rows'] for r in rows],[r['mean_request_itl_ms'] for r in rows],marker='o',label=stage['label'])
    axes[0].set_ylabel('Output tokens/s, mean ± trial SD')
    axes[1].set_ylabel('Mean request ITL (ms)')
    for ax in axes:
        ax.set_xlabel('Logical requests (physical size verified separately)')
        ax.set_xticks([1,4,16,32]);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('TP2 / FP32 SSM / BF16 head · 2048 input tokens\nB1: 129 output; B4/16/32: 1025 output · five trials · includes prefill')
    save(fig,a.output,'tp2-optimization-throughput')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,m in zip(axes,(4,32)):
        for i,stage in enumerate(stages):
            row=stage['fidelity'][str(m)]
            ax.errorbar(i,row['mae'],yerr=np.array([[row['mae']-row['ci95'][0]],[row['ci95'][1]-row['mae']]]),fmt='o',capsize=5)
        ax.set_xticks(range(len(stages)),[s['label'] for s in stages]);ax.set_title(f'Physical M{m}');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Gold logprob MAE vs BF16 / 95% query CI')
    fig.suptitle('Fresh teacher-forced scoring · 256 queries / 10,479 target tokens')
    save(fig,a.output,'tp2-optimization-fidelity')
    plt.close(fig)


if __name__=='__main__':
    main()
