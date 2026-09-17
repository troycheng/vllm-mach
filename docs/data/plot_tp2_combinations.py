"""Default/full combination results, with order blocks shown independently."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_tp2_optimization import save

LABELS = {'control': 'Current configuration', 'empty': 'Empty output',
          'gate': 'Attention gate', 'both': 'Empty + gate'}
COLORS = {'empty': 'C0', 'gate': 'C1', 'both': 'C2'}
ROWS = (4, 16, 24, 32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).with_name('tp2-combinations.json'))
    a = p.parse_args()
    data = json.loads(a.data.read_text())['profiles']
    output = Path(__file__).resolve().parent.parent / 'images'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, profile in enumerate(('default', 'full')):
        group = data[profile]
        for variant in ('empty', 'gate', 'both'):
            for block in (0, 1):
                control = group['decode']['control'][block]['rows']
                rows = group['decode'][variant][block]['rows']
                rates = [100 * (r['mean']/c['mean'] - 1) for r, c in zip(rows, control)]
                axes[0, col].plot(ROWS, rates, color=COLORS[variant],
                    linestyle='-' if block == 0 else '--', marker='o' if block == 0 else '^',
                    label=LABELS[variant] + (' · forward' if block == 0 else ' · reverse'))
            control = group['serving']['control']['points']
            rows = group['serving'][variant]['points']
            rates = [100 * (r['aggregate']['output_throughput_tokens_per_s']/
                            c['aggregate']['output_throughput_tokens_per_s'] - 1)
                     for r, c in zip(rows, control)]
            axes[1, col].plot(ROWS, rates, 'o-', color=COLORS[variant], label=LABELS[variant])
        repeat = group['serving']['both'].get('repeat')
        if repeat:
            control = group['serving']['control']['repeat']['points']
            rates = [100 * (r['aggregate']['output_throughput_tokens_per_s'] /
                            c['aggregate']['output_throughput_tokens_per_s'] - 1)
                     for r, c in zip(repeat['points'], control)]
            axes[1, col].plot(ROWS, rates, '^--', color=COLORS['both'],
                             label='Empty + gate · reverse repeat')
        title = 'Default · FP32 SSM / BF16 head' if profile == 'default' else 'Full · FP16 SSM / owner prefill / NVFP4 head'
        axes[0, col].set(title=title, xlabel='Requests / physical rows',
                         ylabel='Decode-workload throughput change (%)')
        axes[1, col].set(title='HTTP 3000/1000 · ' + ('both repeated in reverse order' if repeat else 'single run per configuration'),
                         xlabel='Concurrency', ylabel='Serving throughput change (%)')
        for row in (0, 1):
            axes[row, col].set_xticks(ROWS)
            axes[row, col].axhline(0, color='gray', linewidth=.8)
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=8)
    fig.suptitle('TP2 empty-output + attention-gate combinations\nEach precision profile uses its own control; five decode trials per order block\nDefault reverse block was interrupted by the full cumulative experiment')
    save(fig, output, 'tp2-combinations-throughput')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    names = ['Default control', 'Default both', 'Full control', 'Full both']
    for ax, m in zip(axes, (4, 32)):
        records = [data[profile]['fidelity'][str(m)][variant]
                   for profile in ('default', 'full') for variant in ('control', 'both')]
        ax.errorbar(range(4), [r['mae'] for r in records],
            yerr=np.array([[r['mae']-r['ci95'][0] for r in records],
                           [r['ci95'][1]-r['mae'] for r in records]]), fmt='o', capsize=5)
        ax.set(xticks=range(4), xticklabels=names, title=f'Physical M{m}',
               ylabel='Gold logprob MAE vs BF16 / 95% query CI')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=.2)
        differences = [data[profile]['fidelity'][str(m)]['both']['max_difference_from_control']
                       for profile in ('default', 'full')]
        ax.text(.02, .98, f'Max change vs own control: default {differences[0]:g}; full {differences[1]:g}',
                transform=ax.transAxes, va='top', fontsize=8)
    fig.suptitle('256 queries / 10,479 target tokens per run\nLogprob requests use BF16 logits; full greedy head is checked separately')
    save(fig, output, 'tp2-combinations-fidelity')


if __name__ == '__main__':
    main()
