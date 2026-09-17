"""Full deployment profile: original main versus accepted plan optimizations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_tp2_optimization import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).with_name('tp2-full-cumulative.json'))
    a = p.parse_args()
    data = json.loads(a.data.read_text())['variants']
    output = Path(__file__).resolve().parent.parent / 'images'
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for col, phase in enumerate(('decode', 'serving')):
        for block in (0, 1):
            style = '-' if block == 0 else '--'
            for index, variant in enumerate(('p0', 'current')):
                group = data[variant][phase][block]
                if phase == 'decode':
                    rows = group['rows']
                    rates = [r['mean'] for r in rows]
                    spread = [r['sd'] for r in rows]
                else:
                    rows = group['points']
                    rates = [r['aggregate']['output_throughput_tokens_per_s'] for r in rows]
                    spread = None
                axes[0, col].errorbar([4,16,24,32], rates, yerr=spread, color=f'C{index}',
                    linestyle=style, marker='o', capsize=3,
                    label=('P0 / main' if variant == 'p0' else 'Accepted plan') + f' · block {block}')
            if phase == 'decode':
                base = [r['mean'] for r in data['p0'][phase][block]['rows']]
                current = [r['mean'] for r in data['current'][phase][block]['rows']]
            else:
                base = [r['aggregate']['output_throughput_tokens_per_s'] for r in data['p0'][phase][block]['points']]
                current = [r['aggregate']['output_throughput_tokens_per_s'] for r in data['current'][phase][block]['points']]
            axes[1, col].plot([4,16,24,32], [100*(c/b-1) for c,b in zip(current,base)],
                linestyle=style, marker='o', color='C1', label=f'Block {block}')
        axes[0, col].set(title='2048/1025 development · five trials ± SD' if phase == 'decode'
                            else '3000/1000 HTTP serving · five waves', ylabel='Output tokens/s')
        axes[1, col].set(ylabel='Change from matched P0 (%)')
        axes[1, col].axhline(0, color='gray', linewidth=.8)
        for row in (0, 1):
            axes[row, col].set(xlabel='Requests / concurrency', xticks=[4,16,24,32])
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=8)
    fig.suptitle('TP2 full: P0 → accepted P1-C + P1-B + P2-A\nSame GPU pair · FP16 SSM / owner prefill / NVFP4 head · forward and reverse blocks')
    save(fig, output, 'tp2-full-cumulative-throughput')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, m in zip(axes, (4,32)):
        values = [data[v]['fidelity'][str(m)] for v in ('p0','current')]
        ax.errorbar([0,1], [v['mae'] for v in values],
            yerr=np.array([[v['mae']-v['ci95'][0] for v in values],
                           [v['ci95'][1]-v['mae'] for v in values]]), fmt='o', capsize=5)
        ax.set(title=f'Physical M{m}', xticks=[0,1], xticklabels=['P0 / main','Accepted plan'],
               ylabel='Gold logprob MAE vs BF16 / 95% query CI')
        ax.grid(axis='y', alpha=.2)
        ax.text(.02,.98, f"Max change from P0: {values[1]['max_difference_from_p0']:g}",
                transform=ax.transAxes, va='top', fontsize=9)
    fig.suptitle('Full profile fidelity · 256 queries / 10,479 target tokens\nTeacher-forced BF16 logits; NVFP4 greedy head checked separately')
    save(fig, output, 'tp2-full-cumulative-fidelity')


if __name__ == '__main__':
    main()
