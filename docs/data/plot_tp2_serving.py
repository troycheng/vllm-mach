"""Plot only matched 3000/1000 serving measurements from the TP2 series."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_tp2_optimization import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).with_name('tp2-serving.json'))
    a=p.parse_args()
    data=json.loads(a.data.read_text())
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for index,stage in enumerate(data['stages']):
        ax.plot([p['concurrency'] for p in stage['points']],
                [p['aggregate']['output_throughput_tokens_per_s'] for p in stage['points']],
                marker='o',label=stage['label'],color=plt.get_cmap('tab20')(index % 20))
    ax.set(xticks=[4,16,24,32],xlabel='Concurrency',ylabel='Output tokens/s',
           title='Matched TP2 serving · BF16 activations/KV/head · FP32 SSM\n3000 input / 1000 output · same frozen prompts · single run per point')
    ax.grid(alpha=.2);ax.legend()
    save(fig,Path(__file__).resolve().parent.parent/'images','tp2-serving-throughput')


if __name__=='__main__':main()
