"""Show remaining costs within TP2 only; NInfer has no matching TP2 trace."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_tp2_optimization import save

root = Path(__file__).resolve().parent
rows = [r for r in json.loads((root/'tp2-fusion-analysis.json').read_text())['stages']['p2a-fused-profile'] if r['rank']==0]
groups = [
    ('GDN recurrence', ('persistent_gdn','gdn_recurrence','gdn_convolution')),
    ('AR / residual / norm', ('allreduce_residual_norm',)),
    ('Independent quant', ('activation_quantization',)),
    ('SwiGLU + quant', ('swiglu_quantization',)),
    ('GDN norm + quant', ('gdn_norm_quantization',)),
    ('BA copies + reduction', ('strided_tensor_copy','bf16_splitk_reduction')),
    ('Attention main kernel', ('attention',)),
]
fig, ax = plt.subplots(figsize=(10,4.5),layout='constrained')
width=.11
for i,(label,keys) in enumerate(groups):
    ax.bar([x+(i-3)*width for x in range(4)],
           [sum(r['categories'].get(k,{}).get('union_ms_per_step',0) for k in keys) for r in rows],
           width,label=label)
ax.set(xticks=range(4),xticklabels=['M1','M4','M16','M32'],ylabel='GPU category time (ms / decode step)',
       title='Remaining TP2 costs after P2-A · rank 0 · three profiled steps\nGEMMs and miscellaneous kernels excluded; bars are not end-to-end ITL')
ax.grid(axis='y',alpha=.2)
ax.legend(fontsize=8,ncol=2)
save(fig,root.parent/'images','tp2-fusion-remaining')
