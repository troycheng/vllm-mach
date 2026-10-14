#!/usr/bin/env python3
"""Render the single MXFP8 benchmark report from its per-concurrency dataset."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from render_performance_docs import themed_svg

ROOT = Path(__file__).resolve().parents[1]
POINTS = [4, 16, 24, 32, 48, 64]


def recheck_variants(data, model):
    main = {v['id']:v for v in data['models'][model]['variants']}
    return [{**v, 'points':main[v['main_variant_id']]['points']} if 'main_variant_id' in v else v
            for v in data['baseline_recheck']['variants'][model]]


def recheck_markdown(data):
    check = data.get('baseline_recheck')
    if not check:
        return []
    lines = ['## Baseline and BF16 GDN recheck', '',
             '2B variants run sequentially on GPU 2 and 4B variants sequentially on GPU 3. '
             'Each uses a fresh service and the same updated native binary, prompts, graph policy '
             'and KV budget. Grouped BF16 GDN and GDN PDL are disabled; production projection '
             'PDL remains unchanged. Ordered BF16 is now the default BF16 backend; '
             '`--no-ordered-bf16-ssm` selects stock GDN. Measurements predate the default change.', '',
             'Worker receipts confirm FP32 state and ready ordered GDN for the baseline, '
             'BF16 state and ready ordered GDN for the main BF16 matrix. Stock measurements '
             'are retained below as an explicit backend comparison. The main five configuration tables above use '
             'this recheck. The earlier raw matrix remains archived.', '']
    previous=check.get('previous_matrix_points')
    if previous:
        lines += ['Change from the preceding matrix baseline (same configuration, different GPU):', '',
                  '| Concurrency | 2B FP32 / BF16 KV | 4B FP32 / BF16 KV |',
                  '|---:|---:|---:|']
        for i,c in enumerate(POINTS):
            gains=[]
            for model in ('2b','4b'):
                old=next(v for v in previous[model] if v['id']=='baseline')['points'][i]['metrics']['output_throughput_tokens_per_s']
                new=next(v for v in recheck_variants(data, model) if v['id']=='baseline')['points'][i]['metrics']['output_throughput_tokens_per_s']
                gains.append(f'{100*(new/old-1):+.2f}%')
            lines.append('| c'+str(c)+' | '+' | '.join(gains)+' |')
        lines += ['', 'The dtype, cache and GDN route checks pass in both runs. Cross-card '
                  'differences above describe observed variability; the state comparisons below '
                  'use the fresh baseline on the same GPU.', '']
    for model in ('2b','4b'):
        variants = {v['id']:v for v in recheck_variants(data, model)}
        ids = ['baseline','stock','ordered']
        labels = ['FP32 / BF16 KV','BF16 stock / BF16 KV','BF16 ordered / BF16 KV']
        if model=='4b':
            ids += ['stock-kv-fp8','ordered-kv-fp8']
            labels += ['BF16 stock / FP8 KV','BF16 ordered / FP8 KV']
        lines += [f"### {model.upper()} route comparison (output tokens/s)", '',
                  '| Concurrency | ' + ' | '.join(labels) + ' |',
                  '|---:|' + '---:|' * len(ids)]
        for i,c in enumerate(POINTS):
            lines.append('| c'+str(c)+' | '+' | '.join(f"{variants[k]['points'][i]['metrics']['output_throughput_tokens_per_s']:.2f}" for k in ids)+' |')
        lines += ['']
    pair=check.get('four_b_fp8_kv_ssm_pair')
    if pair:
        lines += ['### 4B SSM precision with FP8 KV held fixed (GPU 4)', '',
                  'Both fresh services run sequentially on GPU 4. This pair isolates the '
                  'complete FP32 ordered versus BF16 stock configuration at the same KV dtype, '
                  'matching the October 12 SSM study. It uses a different GPU from the main matrix.', '',
                  '| Concurrency | FP32 SSM / FP8 KV | BF16 stock SSM / FP8 KV | BF16 vs FP32 | October 12 gain |',
                  '|---:|---:|---:|---:|---:|']
        historical=data['historical_validation']['ssm_throughput']['models']['4b']['points']
        for i,c in enumerate(POINTS):
            a=pair[0]['points'][i]['metrics']['output_throughput_tokens_per_s']
            b=pair[1]['points'][i]['metrics']['output_throughput_tokens_per_s']
            lines.append(f"| c{c} | {a:.2f} | {b:.2f} | {100*(b/a-1):+.2f}% | {historical[i]['gain_pct']:+.2f}% |")
        lines += ['']
    now4={v['id']:v for v in recheck_variants(data, '4b')}
    old4=data['historical_validation']['ordered_bf16']['serving']['models']['4b']['variants']['previous']
    lines += ['### 4B historical configuration alignment', '',
              'The October 13 previous-ordered run used **BF16 SSM / FP8 E4M3 KV**, '
              'confirmed by its archived launcher argv. It was not a BF16 KV measurement. '
              'The current BF16 KV row is a separate configuration required by the unquantized-KV matrix. '
              'Only the FP8 KV columns below use matching SSM/KV/backend settings; '
              'native revision and GPU still differ between dates.', '',
              '| Concurrency | Historical ordered / FP8 KV | Current ordered / FP8 KV | Change at matching dtype | Current ordered / BF16 KV |',
              '|---:|---:|---:|---:|---:|']
    for i,c in enumerate(POINTS):
        old=old4['points'][i]['metrics']['output_throughput_tokens_per_s']
        fp=now4['ordered-kv-fp8']['points'][i]['metrics']['output_throughput_tokens_per_s']
        bf=now4['ordered']['points'][i]['metrics']['output_throughput_tokens_per_s']
        lines.append(f'| c{c} | {old:.2f} | {fp:.2f} | {100*(fp/old-1):+.2f}% | {bf:.2f} |')
    lines += ['', 'To select BF16 SSM and BF16 KV explicitly, use '
              '`--ssm-dtype bfloat16 --kv-cache-dtype bfloat16`. '
              'To reproduce the historical dtype/backend selection, use '
              '`--ssm-dtype bfloat16 --kv-cache-dtype fp8_e4m3 --ordered-bf16-ssm`. '
              'Grouped BF16 GDN and GDN PDL remain disabled in this previous-ordered comparison.', '']
    now2={v['id']:v for v in recheck_variants(data, '2b')}
    ordered2=now2['ordered']['points'][-1]['metrics']['output_throughput_tokens_per_s']
    stock2=now2['stock']['points'][-1]['metrics']['output_throughput_tokens_per_s']
    old2=data['historical_validation']['ordered_bf16']['serving']['models']['2b']['variants']['previous']['points'][-1]['metrics']['output_throughput_tokens_per_s']
    lines += [f'At c64, 2B ordered BF16 reaches **{ordered2:.2f} tokens/s**, versus '
              f'**{stock2:.2f}** for current stock BF16 and **{old2:.2f}** for the '
              f'October 13 ordered run ({100*(ordered2/old2-1):+.2f}% versus that historical run). '
              'The default backend now selects this ordered configuration; use '
              '`--no-ordered-bf16-ssm` to reproduce stock GDN.', '']
    lines += ['The October 13 GDN study used stock/ordered/grouped BF16 routes, '
              'BF16 KV for 2B and FP8 KV for 4B, with an earlier native binary '
              '(`c4522aff…`). Its runtime source hashes match the current runtime; '
              'native binary, GPU and route differences prevent attributing changes '
              'to SSM precision alone. Ordered BF16 retains the unresolved correctness '
              'limitations described in the [GDN guide](mxfp8-gdn.md).', '']
    return lines


def markdown(data):
    assert data['complete'], 'Complete the five configuration screens first'
    lines = [
        '# MXFP8 serving benchmark', '',
        'Qwen3.5-2B and Qwen3.5-4B on one RTX 5090 per model, TP1. '
        'Each table reports **output tokens/s at an individual request concurrency** '
        'and compares that configuration with the historical open-source **block FP8** reference.', '',
        'Launcher defaults: 4B uses **FP32 SSM / FP8 E4M3 KV** when both dtype flags are omitted. '
        'Changing the default GDN backend to ordered does not change either storage dtype. '
        '2B fixes KV to BF16. Stock vLLM defaults to `auto` (model dtype). '
        'The 4B baseline explicitly passes `--kv-cache-dtype bfloat16`.', '',
        'The baseline is **FP32 SSM state and BF16 KV**, with MXFP8 model weights. '
        'SSM precision reduction means FP32 to BF16; KV quantization means BF16 to E4M3 FP8. '
        'The 4B progression retains BF16 SSM when enabling FP8 KV. The 2B matrix retains BF16 KV.', '',
        'Current MXFP8 measurements use native revision `' + data['native_source_commit'][:7] +
        '` and the same fused SwiGLU/MXFP8 quantization and hybrid greedy-head optimizations. '
        'The production graph and PDL selectors remain active. Both SSM dtypes use ordered '
        'GDN in the main matrix; BF16 rounds dense state after each pending token. '
        'Stock BF16 comparisons remain in the recheck section.', '',
        'Block FP8 values are retained historical open-source measurements, not fresh '
        'paired trials. The 2B reference used another RTX 5090. Gains describe complete '
        'configurations; single-run differences have no repeated-trial confidence interval.', '',
        '[Machine-readable data](data/mxfp8-benchmark.json) retain exact rates, '
        'per-point latency metrics, request counts, artifact hashes and runtime receipts.', '',
    ]
    for model in ('4b', '2b'):
        entry = data['models'][model]
        refs = {p['concurrency']:p['output_tokens_s'] for p in data['block_fp8_references'][model]['points']}
        lines += [f'## Qwen3.5-{model.upper()}', '',
                  f"![Qwen3.5-{model.upper()} per-concurrency throughput](assets/qwen35-{model}-throughput.svg)", '']
        expected = ['baseline','ssm_bf16','ssm_bf16_kv_fp8'] if model == '4b' else ['baseline','ssm_bf16']
        assert [v['id'] for v in entry['variants']] == expected
        for variant in entry['variants']:
            assert [p['concurrency'] for p in variant['points']] == POINTS
            lines += [f"### {variant['label']}", '',
                      '| Concurrency | Open-source block FP8 | MXFP8 | Gain vs block FP8 |',
                      '|---:|---:|---:|---:|']
            for point in variant['points']:
                c = point['concurrency']
                rate = point['metrics']['output_throughput_tokens_per_s']
                gain = 100 * (rate / refs[c] - 1)
                assert abs(gain - point['gain_vs_block_fp8_percent']) < 1e-8
                lines.append(f'| c{c} | {refs[c]:.2f} | {rate:.2f} | {gain:+.2f}% |')
            lines += ['', f"Launch flags: `{' '.join(variant['launch_flags'])}`.", '']
    lines += recheck_markdown(data)
    lines += [
        '## Workload and verification', '',
        '| Concurrency | Scored requests | Input tokens/request | Output tokens/request |',
        '|---:|---:|---:|---:|',
    ]
    for c, count in zip(POINTS,(40,160,120,160,240,320)):
        lines.append(f'| c{c} | {count} | 3000 | 1000 |')
    lines += [
        '', 'Each configuration completes 1,040 scored requests. Request concurrency '
        'is a client limit; physical decode rows can differ because of graph padding '
        'and scheduler tails. Startup, JIT and both warmup phases are excluded.', '',
        'Each point performs independent prewarm (`c` requests with 256 output tokens), '
        'internal warmup (`c` requests with 128 output tokens), then the scored requests. '
        'Uniform prompts use token IDs in `[1000, 240000)`, contract seed `20260924`, '
        'and scored request seed base `2026092400`; independent prewarm uses '
        '`20261003` and `2026100300`. Requests are greedy and streamed, with '
        '`ignore_eos=true`, `top_k=-1`, `top_p=1`, and prefix caching disabled.', '',
        'Output throughput uses successful completion-token usage divided by the '
        'scored duration of that concurrency point. TTFT starts after semaphore admission; '
        'TPOT is `(response end - first token time) / (completion_tokens - 1)`. '
        'Streaming-event intervals are not independently reconstructed model-token timestamps.', '',
        'Every current configuration passes exact request/token accounting and worker '
        'receipt checks for the native library, SSM dtype and actual KV storage. '
        '4B disables the optional `VLLM_SMALL_BATCH_ATTN` experimental branch and uses '
        'standard FlashInfer attention in both KV modes. Both models use a 19 GiB KV '
        'budget and FP32 or BF16 SSM as stated in each table.', '',
        '## Reproduction', '',
        'Build [Dockerfile.mxfp8](../deploy/Dockerfile.mxfp8), prepare the '
        '[4B](mxfp8-model.md) or [2B](mxfp8-2b-model.md) checkpoint, and start '
        'the [4B](mxfp8-champion.md) or [2B](mxfp8-2b-champion.md) launcher with '
        'the flags shown under the relevant table. 2B retains '
        '`--attention-backend TRITON_ATTN --hybrid-head`; 4B enables the hybrid head by default.', '',
        'Against an already-ready server, use a new output directory:', '',
        '```bash',
        'python -m vllm_mach.mxfp8.benchmark \\',
        '  --base-url http://127.0.0.1:8000 --model q35-mx8-study \\',
        '  --outdir ./sixpoint-results',
        '```', '',
        'Use `q35-2b-study` for 2B. The client sends requests and does not manage '
        'the server. `--points 4 16` runs a subset; only the complete six-point '
        'suite receives `complete=true`. Failed or incomplete requests produce '
        'exit code 2 and preserve the raw evidence.', '',
        '## Historical fidelity', '',
        'Earlier fixed-gold, SSM and KV precision records are retained in '
        '`historical_validation` in the same JSON. They identify earlier native '
        'revisions and are not new numerical qualification of this throughput matrix. '
        'The [KV calibration notes](mxfp8-kv.md) and profile guides describe their scope.', '',
        '## Regeneration', '',
        '```bash', 'python tools/render_mxfp8_benchmark.py',
        'python tools/render_mxfp8_benchmark.py --check', '```', '',
    ]
    return '\n'.join(lines)


def plots(data, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none',
                         'svg.hashsalt':'mach-mxfp8-matrix','axes.spines.top':False,
                         'axes.spines.right':False})
    colors = ['#9ba6b7','#9bc4f7','#007bff','#0052ad']
    for model in ('4b','2b'):
        entry = data['models'][model]
        series = [('Historical block FP8',[p['output_tokens_s'] for p in data['block_fp8_references'][model]['points']])]
        labels = {'baseline':'FP32 SSM / BF16 KV','ssm_bf16':'BF16 ordered SSM / BF16 KV','ssm_bf16_kv_fp8':'BF16 ordered SSM / FP8 KV'}
        series += [(labels[v['id']],[p['metrics']['output_throughput_tokens_per_s'] for p in v['points']]) for v in entry['variants']]
        fig,ax = plt.subplots(figsize=(12,4.8))
        width = .8/len(series)
        for j,(label,values) in enumerate(series):
            positions = [i+(j-(len(series)-1)/2)*width for i in range(6)]
            bars = ax.bar(positions,values,width*.9,color=colors[j],label=label)
            ax.bar_label(bars,labels=[f'{v:,.0f}' for v in values],padding=3,fontsize=7)
        ax.set(xticks=range(6),xticklabels=[f'c{c}' for c in POINTS],
               xlabel='Concurrent requests',ylabel='Output tokens/s')
        ax.set_ylim(0,max(v for _,values in series for v in values)*1.15)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.25)
        ax.legend(frameon=False,fontsize=9,ncols=2,loc='upper left',bbox_to_anchor=(0,1.24))
        fig.suptitle(f'Qwen3.5-{model.upper()} — MXFP8 state/KV configurations',fontsize=17,x=.09,ha='left')
        fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.73)
        path = dest/f'qwen35-{model}-throughput.svg'
        fig.savefig(path,metadata={'Date':None});plt.close(fig)
        path.write_text('\n'.join(line.rstrip() for line in themed_svg(path.read_text()).splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    data = json.loads((ROOT/'docs/data/mxfp8-benchmark.json').read_text())
    text = markdown(data)
    with TemporaryDirectory() as tmp:
        dest = Path(tmp);plots(data,dest)
        targets = {ROOT/'docs/mxfp8-benchmark.md':text}
        targets.update({ROOT/'docs/assets'/p.name:p.read_text() for p in dest.glob('*.svg')})
        for path,content in targets.items():
            if args.check:
                assert path.read_text()==content,path
            else:
                path.write_text(content)


if __name__=='__main__':
    main()
