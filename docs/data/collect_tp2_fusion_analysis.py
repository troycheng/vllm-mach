"""Reclassify archived traces for fusion-boundary comparisons, without rerunning models."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from summarize_tp2_decode import summarize_trace


def main():
    raw = ROOT.parent / 'tp2-optimization-20260916'
    stages = {}
    for stage in ('p0-profile', 'p1c-profile', 'p1b-profile',
                  'p2a-fused-off-profile', 'p2a-fused-profile'):
        rows = []
        for m in (1, 4, 16, 32):
            for rank in (0, 1):
                path = raw / stage / f'm{m}-trial0' / f'rank{rank}.json'
                result = summarize_trace(path)
                rows.append(dict(m=m, rank=rank, measured_steps=3,
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    gpu_union_ms_per_step=result['gpu_activity_union_us']/3000,
                    categories={k:dict(count_per_step=v['count']/3,
                        union_ms_per_step=v['union_us']/3000)
                        for k, v in result['categories'].items()}))
        stages[stage] = rows
    ninfer = ROOT.parent / 'ninfer/profiles/bench/mxfp6_tp1/module_comparison.json'
    source_paths = [
        'src/ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh',
        'src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu',
        'src/ops/launcher/rmsnorm.cu', 'src/ops/kernel/rmsnorm.cuh',
        'src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu']
    sources = {name: hashlib.sha256((ROOT.parent/'ninfer'/name).read_bytes()).hexdigest()
               for name in source_paths}
    output = dict(ninfer_source_sha256=sources, schema='mach-tp2-fusion-analysis/v1', stages=stages,
        ninfer_reference=dict(path=str(ninfer), sha256=hashlib.sha256(ninfer.read_bytes()).hexdigest(),
            data=json.loads(ninfer.read_text())),
        limitations=['NInfer reference is TP1 B1, different formats and fused boundaries.',
            'Each rank is independent; category unions must not be added as end-to-end ITL.',
            'BF16 GEMM contains head and BA; native Sm120 GEMMs are classified separately.',
            'Historical profile timings are explanatory, not matched unprofiled speedups.'])
    Path(__file__).with_name('tp2-fusion-analysis.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
