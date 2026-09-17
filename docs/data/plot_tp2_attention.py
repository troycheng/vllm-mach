"""Focused matched-control figures for the P2-B attention producer."""
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
    p.add_argument('--data', type=Path, default=Path(__file__).with_name('tp2-p2b.json'))
    p.add_argument('--validation', type=Path, default=Path(__file__).with_name('tp2-attention-validation.json'))
    p.add_argument('--serving', type=Path, default=Path(__file__).with_name('tp2-attention-serving.json'))
    a = p.parse_args()
    stage = json.loads(a.data.read_text())
    validation = json.loads(a.validation.read_text())
    serving = json.loads(a.serving.read_text())
    output = Path(__file__).resolve().parent.parent / 'images'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for label, data in [('Control', stage['matched_control']), ('Fused gate (probe)', stage)]:
        rows = data['decode']['rows']
        axes[0].errorbar([r['rows'] for r in rows], [r['mean_output_tokens_per_s'] for r in rows],
            yerr=[r['stdev_output_tokens_per_s'] for r in rows], marker='o', capsize=4, label=label)
    axes[0].set(title='2048-token input · five trials ± SD', xlabel='Requests / physical rows',
                ylabel='Output tokens/s', xticks=[1, 2, 4, 8, 16, 24, 32])
    for run in serving['stages']:
        axes[1].plot([p['concurrency'] for p in run['points']],
                     [p['aggregate']['output_throughput_tokens_per_s'] for p in run['points']],
                     marker='o', label=run['label'])
    axes[1].set(title='Matched HTTP 3000/1000 · single run', xlabel='Concurrency',
                ylabel='Output tokens/s', xticks=[4, 16, 24, 32])
    primary = [next(r for r in data['decode']['rows'] if r['rows'] == 32)
               for data in (stage['matched_control'], stage)]
    reverse = [next(r for r in validation['reverse_order'] if r['mode'] == mode)
               for mode in ('off', 'on')]
    for group, pair in enumerate((primary, reverse)):
        base = pair[0]['mean_output_tokens_per_s']
        for i, (label, color, offset) in enumerate([('Control', 'C0', -.08), ('Fused gate', 'C1', .08)]):
            row = pair[i]
            axes[2].errorbar(group + offset, 100 * row['mean_output_tokens_per_s'] / base,
                yerr=100 * row['stdev_output_tokens_per_s'] / base, fmt='o', capsize=4,
                color=color, label=label if group == 0 else None)
    axes[2].axhline(100, color='gray', linewidth=.8)
    axes[2].set(xticks=[0, 1], xticklabels=['GPUs 4/5\non → off', 'GPUs 6/7\noff → on'],
                title='Independent M32 order check', ylabel='% of each pair’s control ± trial SD')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('P2-B attention gate → MXFP8 · TP2 / BF16 head / FP32 SSM')
    save(fig, output, 'tp2-attention-throughput')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    rows = [stage['fidelity'][str(m)] for m in (4, 32)]
    axes[0].errorbar([4, 32], [r['mae'] for r in rows],
        yerr=np.array([[r['mae']-r['ci95'][0] for r in rows],
                       [r['ci95'][1]-r['mae'] for r in rows]]), fmt='o', capsize=5)
    axes[0].set(title='Error vs original BF16 model', ylabel='Gold logprob MAE / 95% query CI')
    diffs = [validation['fidelity'][str(m)]['max_difference_from_accepted'] for m in (4, 32)]
    axes[1].plot([4, 32], diffs, 'o')
    for m, v in zip((4, 32), diffs):
        axes[1].annotate(f'{v:g}', (m, v), xytext=(0, 8), textcoords='offset points', ha='center')
    axes[1].set(title='Difference vs accepted P2-A', ylabel='Maximum gold logprob difference')
    if max(diffs) == 0:
        axes[1].set_ylim(-.001, .001)
    for ax in axes:
        ax.set(xticks=[4, 32], xlabel='Physical rows'); ax.grid(alpha=.2)
    fig.suptitle('Fresh fidelity · 256 queries / 10,479 target tokens per size')
    save(fig, output, 'tp2-attention-fidelity')


if __name__ == '__main__':
    main()
