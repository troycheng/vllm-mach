# Dense producer fusion

Eligible Dense TP2 decode uses the existing MXFP6 producer operations for
rounded SwiGLU quantization and gated RMSNorm/output projection. The required
extension source revision is
[`cd4e964c391fcb8aaf1a27d28a63d778e3a38ece`](https://github.com/Nekofish-L/mxfp6_sm120/commit/cd4e964c391fcb8aaf1a27d28a63d778e3a38ece).

The running port is limited to `gdn_decode.py`, `gdn_output.py`,
`fused_mlp.py`, and `warmup.py`. Warmup prepares the two optional routes. The
GDN route combines gated RMSNorm and the MXFP8 output projection. The MLP route
uses the extension producer after the existing TP2 gate/up projection. Both
keep the pre-existing implementation for unsupported input shapes and models.
MoE remains on its existing path.

## Admission

`VLLM_MACH_FUSED_GDN_QUANT` and `VLLM_MACH_FUSED_SWIGLU_QUANT` each accept
`auto`, `0`, or `1`; `auto` is the default. `0` disables its route. `auto`
selects it for native Dense TP2 geometry. Both installation and startup
require the new extension operations: an older 0.2.1 binary is rejected,
without silently disabling fusion. `1` also explicitly requires the operation
during eligible layer preparation. Ineligible model geometry continues on
its existing model implementation. Runtime fusion is
limited to BF16 CUDA decode rows 1/2/4/8/16/24/32 and retains the existing
fallback for other shapes.

## Current bounded evidence

The offline comparison used Qwen3.8-27B native MXFP6 on two RTX 5090 GPUs,
TP2, vLLM 0.29.0, FP32 SSM, a BF16 head, and 8218214400 bytes of KV cache per
rank. It used 64 simultaneous offline LLM requests with 3000 input and 1000
output tokens, scheduler concurrency 32, full decode graphs, and a complete
64x1000-output-token warmup per arm. Accepted then base order used two scored
timings per arm; the matched gains were 1.6931% and 1.7518% (pooled 1.7225%). No scored JIT
compilations were recorded. The baseline is Mach `1a2c67e` with its original
MXFP6 extension; the candidate includes scale initialization in the updated
extension as well as the two Mach routes. Merely disabling the two Mach flags
on the new extension does not restore this original baseline.

| Scored timing | Base output tok/s | Fused output tok/s | Gain |
|---|---:|---:|---:|
| 1 | 1517.7572 | 1543.4551 | +1.6931% |
| 2 | 1515.4418 | 1541.9888 | +1.7518% |
| Total tokens / total elapsed time | 1516.5987 | 1542.7216 | +1.7225% |

This is an offline LLM measurement; the README's existing HTTP comparisons
and NVFP4 measurements use different contracts and remain unchanged.

For 256 queries and 10479 target tokens, both physical M4 and M32 runs had
zero per-token gold-logprob delta between base and accepted. Their query-mean
MAE against the matching archived 2026-09-17 BF16 reference was respectively
0.0855531417 and 0.0913294934. BF16 was not rerun for this comparison.

The producer implementations come from the team's `dev-ninfer` P1-C,
P1-B and P2-A work. This integration preserves the current workspace setup
and geometry-aware GDN fallback. The separate model results above cover the same computation paths. The
publication update additionally rejects missing extension operations during
installation, startup, and layer preparation.

[Machine-readable summary](data/dense-producer-fusion-20260919.json) records
only the public aggregate contract and results.
