# FP16 recurrent state and BA scheduling

Optional configuration for Qwen3.8-27B K5/K6 EXL3 with checkpoint MXFP6, vLLM 0.28.0, TP2 and SM120. It stores GDN recurrent state in FP16; convolution state and attention KV remain BF16. Existing profiles retain their previous state dtype and scheduling.

FP16 recurrent storage changes numerical precision. Check your workload before enabling it. The implementation uses the existing packed recurrent decode and Triton/FLA prefill paths. FlashInfer specialized decode still requires FP32 recurrent state and falls back for this profile. This change does not introduce an FP16 FlashInfer kernel.

## Configuration

Install the matching Mach wheel and reapply the [GDN overlay](../profiles/flashinfer-0.6.18-gdn/README.md). The installer accepts the released a8 overlay as an upgrade source and rejects unknown local edits. Keep the dependencies and native extensions used by the [long-prefill profile](long-prefill.md).

```bash
export VLLM_MACH_MXFP6_CHECKPOINT=/path/to/paired-mxfp6-checkpoint
source profiles/vllm-0.28.0/qwen38-checkpoint-fp16-ssm.env
```

Add `--mamba-ssm-cache-dtype float16` to the existing serve command. The environment flag alone does not change cache allocation. Keep capture sizes `[1,2,4,8,16,24,32]`, BF16 model/attention KV, TP2 and non-speculative decoding.

`VLLM_MACH_FP16_SSM_STATE=1` admits FP16 state only for hidden size 5120, 16 K heads, 48 V heads, K/V head dimension 128, non-interleaved GQA and packed recurrent decode on SM120. It does not broaden speculative decoding support.

`VLLM_MACH_BA_OVERLAP_M16_M24=1` adds physical M16/M24 to the BA scheduler when FP16 state is admitted. It also requires `VLLM_MACH_BA_OVERLAP=1`. These rows retain EXL3 QKV; the checkpoint route still uses merged MXFP6 QKV at M32. BA runs on an auxiliary stream and joins before packed recurrence. Disabling the extra-row flag leaves M32 scheduling available. Disabling the master BA flag disables all BA overlap.

The supplied profile enables these switches and reuses Temporal M24 and the 32-shape prefill configuration. Neither the base package nor the previous profiles enable FP16 state automatically.

## Source evidence

The September 9 K5/K6 experiment measured 386.01 / 1066.85 / 1329.31 / 1564.36 output tokens/s at c4/c16/c24/c32, with 3000 input and 1000 output tokens. Adding M16/M24 BA to its FP16-state parent improved c16/c24 by 1.69%/1.45%; c4/c32 were essentially unchanged. This is a single complete source-stack run compared with an earlier same-contract run, not a measured speedup over released Mach.

The source's matched BF16 check used 256 queries and 10,479 target tokens at physical M32: mean per-query raw-logprob MAE was 0.091515. This is a numerical metric, not a task error rate. The scheduling change separately passed 1728 output/conv/state byte comparisons on both ranks. It did not repeat model quality testing and does not establish long-context fidelity.

## Mach validation

The development wheel passed 169 CPU tests, including opt-in and dtype guards, unsupported configurations, FI fallback, M16/M24/M32 routing and a8 overlay upgrades.

TP2 acceptance on two RTX 5090 GPUs verified FP16 recurrent state and BF16 convolution state at all 48 GDN layers per rank. Serial and overlapping execution matched in all 2592 output/conv/SSM byte comparisons across M16/M24/M32, three changed-input cases and two recurrence steps. State restoration, canary slots, pending cleanup and temporary Graph-buffer cleanup also passed.

A fresh service passed all 40 task checks with zero pass/fail regressions; 37/40 outputs matched the stored reference text exactly. It also completed 32 requests each at c16/c24/c32 with 3000 input and 128 output tokens. These are bounded integration checks, not a matched full-throughput comparison or a repeat of the source BF16 logprob evaluation. The tested Mach image used ExLlamaV3 1.4.8; the source experiment used 1.4.6. Source performance and quality numbers are not presented as newly measured Mach results.
