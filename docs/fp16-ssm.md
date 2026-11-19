# FP16 recurrent state and BA scheduling

Optional recurrent-state storage for the Dense and MoE native MXFP6 profiles on TP2 SM120. FP16 stores GDN state at lower precision; convolution state and attention KV remain BF16. FP32 remains the launcher default.

FP16 recurrent storage changes numerical precision. Check your workload before enabling it. The implementation uses the existing packed recurrent decode and Triton/FLA prefill paths. FlashInfer specialized decode still requires FP32 recurrent state and falls back for this profile. This change does not introduce an FP16 FlashInfer kernel.

## Configuration

Build the native MXFP6 image from the [installation guide](installation.md#build-an-image). Add `--fp16-ssm` to `vllm-mach-serve` for the supported Dense or MoE checkpoint. The launcher selects FP16 recurrent storage while retaining BF16 convolution state and attention KV, TP2 and non-speculative decoding. Hybrid head is enabled by default; `--no-hybrid-head` independently selects the complete BF16 head.

The model-specific GDN implementation admits the supported Dense and MoE geometries and retains its existing fallback and state-lifecycle guards. Persistent low-row decode and BA overlap have independent opt-outs; see [GDN decode options](gdn-decode.md). The current installer applies the packaged native source profile.

## Historical source evidence · vLLM 0.28

The September 9 K5/K6 experiment measured 386.01 / 1066.85 / 1329.31 / 1564.36 output tokens/s at c4/c16/c24/c32, with 3000 input and 1000 output tokens. Adding M16/M24 BA to its FP16-state parent improved c16/c24 by 1.69%/1.45%; c4/c32 were essentially unchanged. This is a single complete source-stack run compared with an earlier same-contract run, not a measured speedup over released Mach.

The source's matched BF16 check used 256 queries and 10,479 target tokens at physical M32: mean per-query raw-logprob MAE was 0.091515. This is a numerical metric, not a task error rate. The scheduling change separately passed 1728 output/conv/state byte comparisons on both ranks. It did not repeat model quality testing and does not establish long-context fidelity.

## Historical EXL3 validation

The development wheel passed 169 CPU tests, including opt-in and dtype guards, unsupported configurations, FI fallback, M16/M24/M32 routing and a8 overlay upgrades.

TP2 acceptance on two RTX 5090 GPUs verified FP16 recurrent state and BF16 convolution state at all 48 GDN layers per rank. Serial and overlapping execution matched in all 2592 output/conv/SSM byte comparisons across M16/M24/M32, three changed-input cases and two recurrence steps. State restoration, canary slots, pending cleanup and temporary Graph-buffer cleanup also passed.

A fresh service passed all 40 task checks with zero pass/fail regressions; 37/40 outputs matched the stored reference text exactly. It also completed 32 requests each at c16/c24/c32 with 3000 input and 128 output tokens. These are bounded integration checks, not a matched full-throughput comparison or a repeat of the source BF16 logprob evaluation. The tested Mach image used ExLlamaV3 1.4.8; the source experiment used 1.4.6. Source performance and quality numbers are not presented as newly measured Mach results.
