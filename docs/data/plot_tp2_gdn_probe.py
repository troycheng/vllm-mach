"""Plot repaired producer throughput and exactness separately from model data."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'mach-gdn-probe'})
data=json.loads((root/'tp2-gdn-quant-probe.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for rank,run in enumerate(data['fixed_runs']):
    rows=[r for r in run['results'] if r['weight_dtype']=='torch.bfloat16']
    m=[r['rows'] for r in rows]
    for field,label in [('reference_ms','Existing'),('candidate_ms','Fixed fused producer')]:
        axes[0].plot(m,[r['rows']*1000/statistics.mean(r[field]) for r in rows],marker='o',label=f'{label}, GPU {data["fixed_devices"][rank]}')
    if rank == 0:
        axes[1].plot(m,[sum(c['norm_mismatches'] for c in r['checks']) for r in rows],marker='o',label='Fixed explicit layout')
        old=[r for r in data['runs'][0]['results'] if r['weight_dtype']=='torch.bfloat16']
        axes[1].plot(m,[sum(c['norm_mismatches'] for c in r['checks']) for r in old],marker='x',label='Initial automatic layout')
axes[0].set_ylabel('Local producer rows/s (CUDA Graph microbenchmark)')
axes[1].set_ylabel('BF16 norm values differing from existing producer')
for ax in axes:
    ax.set_xlabel('Physical M (24 local heads)');ax.set_xticks([1,4,8,16,24,32]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('P2-A fused producer repaired: exact BF16 intermediate + packed MXFP8\nLocal operator throughput; model measurements reported separately')
from plot_tp2_optimization import save
save(fig,root.parent/'images','tp2-gdn-quant-probe')
