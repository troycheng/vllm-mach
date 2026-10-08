"""Produce a reviewable TP1 report from fixed-token profiles and prior evidence."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


def table(headers, rows):
    return ('<table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
                f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows)
            + '</tbody></table>')


def fidelity(root):
    def load(path):
        return {r['id']: np.asarray(r['gold_logprobs']) for r in json.loads(path.read_text())}

    a = load(root / 'fidelity-baseline/records.json')
    b = load(root / 'fidelity-norm_quant_pdl/records.json')
    ref = load(root.parent / 'mxfp8-tp1-fidelity/m32/bf16/records.json')
    assert a.keys() == b.keys() == ref.keys()
    before = np.array([np.abs(a[k] - ref[k]).mean() for k in ref])
    after = np.array([np.abs(b[k] - ref[k]).mean() for k in ref])
    delta = after - before
    rng = np.random.default_rng(20260930)
    return {
        'queries': len(a), 'tokens': sum(map(len, ref.values())), 'physical_batch': 32,
        'gpu': 7, 'baseline_query_mae_vs_bf16': float(before.mean()),
        'norm_quant_pdl_query_mae_vs_bf16': float(after.mean()),
        'paired_delta': float(delta.mean()),
        'paired_delta_ci95': np.quantile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1), [.025, .975]).tolist(),
        'baseline_repeat': json.loads((root / 'fidelity-baseline/repeat.json').read_text()),
        'norm_quant_pdl_repeat': json.loads((root / 'fidelity-norm_quant_pdl/repeat.json').read_text()),
        'note': 'Frozen256 short-context teacher-forced diagnostic; not the 3000/1000 workload. BF16 reference reused from earlier measurements.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--long-window', type=Path)
    parser.add_argument('--pdl-complete', type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    data['fidelity'] = fidelity(args.audit)
    data['native_library'] = json.loads((args.audit / 'native-library.json').read_text())
    previous = json.loads(Path('docs/data/mxfp8-tp1-breakdown-20260930.json').read_text())
    data['historical_prefill_profiles'] = previous['prefill_profiles']
    data['historical_comparison_source'] = 'docs/data/mxfp8-tp1-breakdown-20260930.json'
    data['prior_thread'] = '01a0f130-a267-7833-96df-18b2888148bc'
    data['tests'] = {'new_producer_and_compile_cache': 11, 'regression_distinct_passed': 72,
                     'total_distinct_passed': 83,
                     'note': 'The native workspace suites need separate processes: a combined run froze the pool before a later suite resized it; the affected suite passed all 3 checks in a fresh process.'}
    profiles = data['profiles']
    breakdown, serving, changes, prefill = [], [], [], []
    for bs in (1, 16, 32):
        base = profiles[f'default_bs{bs}']
        breakdown.append([bs, *(f'{base["groups_ms"][k]:.3f}' for k in ('GEMM', 'full_attention', 'GDN', 'other')),
                          f'{base["kernel_sum_ms"]:.3f}', f'{base["gpu_span_median_ms"]:.3f}'])
        for arm in ('default', 'pdl', 'norm_quant', 'norm_quant_pdl', 'nvfp4_head'):
            p = profiles[f'{arm}_bs{bs}']
            s = p['serving']
            samples = ' / '.join(f'{v:.3f}' for v in s['tpot_samples_ms'])
            serving.append([bs, arm, f'{s["mean_tpot_ms"]:.3f}', samples,
                            f'{s["output_throughput_tokens_s"]:.1f}', f'{s["mean_ttft_ms"]:.1f}'])
            checks = data['output_smoke_checks'][f'{arm}_bs{bs}']
            changes.append([bs, arm, f'{100 * (1 - s["mean_tpot_ms"] / base["serving"]["mean_tpot_ms"]):+.2f}%',
                            f'{p["gpu_span_median_ms"]:.3f}', int(p['kernels_per_step']),
                            f'{checks["text_matches_vs_default"]}/{checks["text_comparisons_vs_default"]}',
                            f'{checks["text_matches_repeated_runs"]}/{checks["text_comparisons_repeated_runs"]}'])
        old = data['historical_prefill_profiles'][f'default_bs{bs}']
        prefill.append([bs, old['input_tokens'], old['steps'],
                        *(f'{old["total_groups_ms"][k]:.1f}' for k in ('GEMM', 'full_attention', 'GDN', 'other'))])
    detail_names = sorted({k for b in (1, 16, 32) for k in profiles[f'default_bs{b}']['details_ms']})
    details = [[k, *(f'{profiles[f"default_bs{b}"]["details_ms"].get(k, 0):.4f}' for b in (1, 16, 32))] for k in detail_names]
    doc = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MXFP8 TP1 3000/1000</title>
<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#202b3d;background:#fafbfe}h1{font-size:30px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:9px 12px;border-bottom:1px solid #dce2eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf1f9}pre{white-space:pre-wrap;padding:18px;background:#edf1f9;border-radius:8px}p{line-height:1.7}img{width:100%}.note{border-left:4px solid #4263eb;padding-left:16px}</style>
<h1>Qwen3.5-4B-MXFP8 · TP1 · 3000 输入 / 1000 输出</h1>
<p>2026-09-30，RTX 5090 GPU 2，native MXFP8 使用 mxfp6 包，BF16 KV/SSM，Triton attention，默认 VLLM_COMPILE / FULL_AND_PIECEWISE。32 层：24 GDN + 8 full attention。GDN TP1 优化未开启。</p>
<p class="note">每个 BS 每组配置先完成一次同 BS 的 3000/1000 预热，再运行三次未开启 profiler 的 3000/1000 请求。另一次 3000/1000 请求延迟 80 个调度 iteration 后采样 40 个固定 BS decode step。分项数据是此早期窗口的 kernel duration 总和，不是整个 1000 输出的平均耗时；attention 随上下文增长。HTTP TPOT 覆盖完整输出，包含最初 mixed prefill、队尾及前端开销。不同请求同步到达，服务并发为 BS；trace 校验纯 decode 时实际 BS。</p>
'''
    if args.pdl_complete:
        complete = json.loads(args.pdl_complete.read_text())
        data['pdl_complete'] = {'source': str(args.pdl_complete), **complete}
        rows = []
        for bs in (1, 16, 32):
            baseline = complete['profiles'][f'default_bs{bs}']
            for arm in ('default', 'pdl', 'norm_quant', 'norm_quant_pdl', 'nvfp4_head_pdl'):
                p = complete['profiles'][f'{arm}_bs{bs}']
                checks = complete['output_smoke_checks'][f'{arm}_bs{bs}']
                s = p['serving']
                gain = 100 * (1 - s['mean_tpot_ms'] / baseline['serving']['mean_tpot_ms'])
                rows.append([bs, arm, f'{s["mean_tpot_ms"]:.3f}', f'{gain:+.2f}%',
                             f'{p["gpu_span_median_ms"]:.3f}',
                             f'{p["groups_ms"]["other"]:.3f}',
                             f'{checks["text_matches_vs_default"]}/{checks["text_comparisons_vs_default"]}'])
        doc += '<h2>最新：补齐非 attention/GDN PDL 后的同库对照（v3）</h2>'
        doc += '<p class="note">此表优先于下方旧版 v1 的 PDL 结论。GPU 2、TP1、BS1/16/32，每组预热后运行三次完整 3000/1000，再采样 40 步。各组均使用同一对新库；仅切换 PDL / norm_quant 开关。启用了 CUTLASS GDC，并补齐 dense cooperative / ping-pong 的 SM120 producer trigger、cache_hint / occupancy / irregular / tma256 的 launch/wait/trigger、原生量化消费者和 GemmaRMSNorm。SwiGLU/量化与残差/norm/量化 producer 也支持 PDL。attention、GDN、BF16 BA 和 lm_head 仍为普通边界。</p>'
        gain_strings = {}
        for arm in ('pdl', 'norm_quant_pdl', 'nvfp4_head_pdl'):
            values = [100 * (1 - complete['profiles'][f'{arm}_bs{bs}']['serving']['mean_tpot_ms']
                            / complete['profiles'][f'default_bs{bs}']['serving']['mean_tpot_ms'])
                      for bs in (1, 16, 32)]
            gain_strings[arm] = ' / '.join(f'{value:.2f}%' for value in values)
        doc += '<p>本轮完整请求 TPOT 降低（BS1/16/32）：PDL 单独 ' + gain_strings['pdl'] + '；norm_quant + PDL ' + gain_strings['norm_quant_pdl'] + '；已有 NVFP4 head + PDL（norm_quant=0）' + gain_strings['nvfp4_head_pdl'] + '，为本轮最快配置。仅 PDL 的 147 个请求全部与基线文本一致，融合 + PDL 的全部 147 个请求与融合单独开启一致。融合和 NVFP4 head 的文本差异见下表。</p>'
        doc += table(['BS', '配置', '完整请求 TPOT ms', 'TPOT 降低', 'GPU span ms', '其他 ms/step', '相对基线文本相同'], rows)
        latest_breakdown = []
        for bs in (1, 16, 32):
            p = complete['profiles'][f'default_bs{bs}']
            latest_breakdown.append([bs, *(f'{p["groups_ms"][k]:.3f}' for k in ('GEMM', 'full_attention', 'GDN', 'other')),
                                     f'{p["gpu_span_median_ms"]:.3f}'])
        doc += '<p>同版库的基线分解（40-step 早期窗口，ms/step）：</p>'
        doc += table(['BS', 'GEMM', 'full attention', 'GDN', '其他', 'GPU span'], latest_breakdown)
        doc += '<img src="images/mxfp8-tp1-pdl-complete-20260930.png" alt="Complete PDL matched profiles">'
        doc += '<p>默认仍关闭两个实验开关。PDL 允许消费者提前启动，消费者 wait 保证前序写入完成和可见。分项 kernel duration 包含 wait 和重叠，须同时看 GPU span 与未采样完整请求 TPOT。例如 PDL 单独开启时，BS16 的“其他”由约 0.29 增至 1.42 ms，主要是已被重叠隐藏的等待；GPU span 却由约 5.90 降至 5.76 ms，不能把 1.42 ms 当作新增串行开销。</p>'
        doc += '<p>本轮新库验证：6 项 producer/依赖链检查、31 项 Gemma norm、8 项缓存开关、24 项部署和 3 项 native GEMM，共 72 项通过。多段 GEMM→norm→量化/GEMM→SwiGLU/量化→GEMM 链在 M1/16/32 的 changed-input/zero-input CUDA Graph 回放中逐值一致；融合 norm_quant 仍有舍入差异。旧版 fidelity 诊断属于 v1，不视为新库的任务准确率测量。</p>'
        doc += '<p><a href="data/' + args.pdl_complete.name + '">v3 JSON</a> · <a href="data/' + args.pdl_complete.with_suffix('.csv').name + '">v3 CSV</a>。新库、完整 source/patch、SASS 检查和原始 trace：audit/mxfp8-tp1-pdl-complete-20260930。</p>'
    doc += '<h2>历史 v1：基线及优化对照（ms/step）</h2>'
    gains = [100 * (1 - profiles[f'nvfp4_head_bs{bs}']['serving']['mean_tpot_ms']
                    / profiles[f'default_bs{bs}']['serving']['mean_tpot_ms']) for bs in (1, 16, 32)]
    doc += '<p>结论：保留 native MXFP8 与 Triton attention。已有 --nvfp4-lm-head 的完整请求 TPOT 改善为 ' + ' / '.join(f'{g:.2f}%' for g in gains) + '（BS1/16/32），本轮收益最大。新增 norm_quant 每步减少 32 个 kernel、约 0.04 ms，但 HTTP 改善仅 0.2–1.1%；旧版 v1 的 PDL 单独开启无收益，但其 GDC 尚未完整接入，不能作为完整 PDL 链路的结论。融合 + PDL 在 BS1 为约 1.5% 改善，在 BS16/32 未超过融合单独开启。新开关均保持默认关闭。</p>'
    doc += table(['BS', 'GEMM', 'full attention', 'GDN', '其他', 'kernel 合计', 'GPU span'], breakdown)
    doc += '<p>GEMM 含 128 个 MXFP8 投影、24 个 BF16 BA 投影和 lm_head；full attention / GDN 均不重复包含投影。其他含残差/RMSNorm、激活量化、SwiGLU、embedding、sampling 及调度 kernel。GPU span 为第一个 kernel 至最后一个 kernel，kernel 合计与跨 stream 并集不同，不能当作 HTTP TPOT。</p>'
    doc += '<img src="images/mxfp8-tp1-3000-1000-20260930.png" alt="GPU breakdown and spans">'
    doc += '<h2>完整 3000/1000 服务对照</h2>' + table(['BS', '配置', '平均 TPOT ms', '三次 TPOT ms', '输出 tokens/s', '平均 TTFT ms'], serving)
    if args.long_window:
        long_data = json.loads(args.long_window.read_text())
        data['long_window'] = {'source': str(args.long_window),
                               'gpu': long_data['cuda_visible_devices'],
                               'contract': long_data['contract'],
                               'profiles': long_data['profiles']}
        long_rows = []
        for bs in (1, 16, 32):
            p = long_data['profiles'][f'default_bs{bs}']
            assert p['steps'] == 900
            long_rows.append([bs, p['steps'], *(f'{p["groups_ms"][k]:.3f}' for k in ('GEMM', 'full_attention', 'GDN', 'other')),
                              f'{p["kernel_sum_ms"]:.3f}'])
        doc += '<h2>900-step 稳态长窗口分解（GPU 7，ms/step）</h2>' + table(['BS', 'decode steps', 'GEMM', 'full attention', 'GDN', '其他', '合计'], long_rows)
        doc += '<p>同一 3000/1000 请求在延迟 80 iterations 后测量 900 个固定 BS decode step，覆盖约 90% 输出。该独立 GPU 7 基线更接近整个 decode 的平均工作量；与 GPU 2 的 40-step 优化对照分开，不作跨卡 speedup 计算。初始 mixed prefill 和最终 tail 不在这个固定 BS 分解内。</p>'
    doc += '<h2>优化与输出检查</h2>' + table(['BS', '配置', 'TPOT 降低', 'GPU span ms', 'kernels/step', '相对基线文本相同', '重复文本相同'], changes)
    doc += '<p>norm_quant 仅针对 TP1 dense Qwen3.5 的 32 个 MLP，在 M≤32 时将残差相加、GemmaRMSNorm 和 MXFP8 量化合为一个 producer，直接给 gate/up GEMM 使用；prefill M&gt;32 保留原路径。PDL 使用原生库新增的 gemm_pdl / gemm_from_float_pdl 和带依赖等待的 Triton producer。这些 v1 结果只启用了部分 launch 参数和 Triton producer；CUTLASS GDC 编译开关未开启，自定义 dispatch 尚未接入，因此不代表完整 PDL 链路。</p>'
    doc += '<p>两项新开关默认关闭。NVFP4 head 是已有的候选筛选 + BF16 精算，另行实测；这里没有把其数值行为等同于完整 BF16 head。三次重复不足以证明百分之一左右的改善长期稳定；greedy 文本相同也不能替代准确性评估。</p>'
    f = data['fidelity']
    doc += '<h2>数值验证</h2><p>native PDL GEMM 在五种投影、M1/16/32/128 的 changed-input/zero-input graph 检查中与原路径逐值一致。norm_quant 可能在 BF16 舍入边界改变输出。11 项新测试与 cache 检查、72 项既有回归检查通过（83 个不同检查）；native workspace 套件在独立进程运行。</p>'
    doc += '<p>GPU 7 的 Frozen256、physical M32、' + str(f['tokens']) + ' 个 teacher-forced token：相对历史 BF16 的 per-query gold logprob MAE，匹配 baseline ' + f'{f["baseline_query_mae_vs_bf16"]:.6f}' + '，norm_quant+PDL ' + f'{f["norm_quant_pdl_query_mae_vs_bf16"]:.6f}' + '；差值 ' + f'{f["paired_delta"]:+.6f}' + '，paired bootstrap 95% CI ' + str(f['paired_delta_ci95']) + '。这是短上下文数值诊断，不能直接推断 3000/1000 任务准确率。</p>'
    doc += '<h2>基线组件细分（ms/step）</h2>' + table(['组件', 'BS1', 'BS16', 'BS32'], details)
    doc += '<h2>复用上个会话的 prefill 测量（每个完整 batch 的 GPU kernel 总量，ms）</h2>' + table(['BS', '输入 tokens', 'chunk steps', 'GEMM', 'full attention', 'GDN', '其他'], prefill)
    old_gemm = '、'.join('/'.join(f'{previous["profiles"][f"{arm}_bs{bs}"]["details_ms"]["GEMM/MXFP8_projections"]:.3f}' for arm in ('default', 'b12x')) for bs in (1, 16, 32))
    doc += '<p>Prefill 来自此前 3000 输入 / 1 输出的独立 trace，未在本轮重测。不是每个 chunk 的均值，也不是 TTFT。此前 native / B12x 的 MXFP8 GEMM 为 ' + old_gemm + ' ms（BS1/16/32，旧窗口），未显示切换优势。此前 GDN TP1 只有小幅 GPU span 收益且 BS16 重复文本存在差异；保持实验性。</p>'
    library_root = '/data/lxy/detailed_benchmark/vllm-mach/audit/mxfp8-tp1-pdl-complete-20260930/libraries'
    command = "\n".join([
        f"export MXFP6_LIBRARY_PATH={library_root}/mxfp6_torch.so",
        f"export MXFP8_LIBRARY_PATH={library_root}/mxfp8_torch.so",
        (".venv/bin/python tools/profile_mxfp8_breakdown.py "
        "--output audit/mxfp8-tp1-pdl-complete-repeat "
        "--arms default pdl norm_quant norm_quant_pdl nvfp4_head_pdl "
        "--device 2 --port 8253 --repeats 3 "
        "--output-tokens 1000 --warmup-tokens 1000"),
    ])
    doc += '<h2>复现完整 PDL 对照</h2><pre>' + html.escape(command) + '</pre>'
    doc += '<p>PDL 需要两份重建的原生 .so，并检查 pdl_version≥3。当前源码拒绝不完整旧库；原有 v1 库和四份 Python producer 源码已单独保存。仅融合设置 VLLM_MACH_MXFP8_NORM_QUANT=1；PDL 设置 VLLM_MACH_MXFP8_PDL=1。已有 head 开关为 --nvfp4-lm-head。新库哈希、CUDA/CUTLASS patch、SASS 和源码归档随 v3 audit 保存。</p>'
    doc += '<p><a href="data/' + args.summary.name + '">全部机器可读结果</a> · <a href="data/' + args.summary.with_suffix('.csv').name + '">分项 CSV</a> · <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html">NVIDIA PDL 依赖与 CUDA Graph 文档</a>。原始 trace、逐请求 usage、文本 hash、启动配置和 logprob 记录位于 audit/mxfp8-tp1-3000-1000-20260930。参考会话：01a0f130-a267-7833-96df-18b2888148bc。</p></html>'
    args.summary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    args.html.write_text(doc)


if __name__ == '__main__':
    main()
