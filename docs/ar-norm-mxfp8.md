# Dense AR / RMSNorm / MXFP8 fusion

The Dense profile can produce MXFP8 activations inside its existing tensor-parallel AllReduce/residual/GemmaRMSNorm kernel. This removes the separate activation-quantization launch before GDN QKV and MLP gate/up projections. The kernel retains the original AllReduce, residual and RMSNorm operation order, including BF16 rounding before MXFP8 conversion. It uses the existing official MXFP6 `gemm_w6a8_pdl` consumer; no additional checkpoint or dependency fork is needed.

## Installation and dispatch

Build `native/ar_norm` with the serving environment's CUDA 13.0, PyTorch and FlashInfer 0.6.18, then install Mach. The default `deploy/install.py` and Docker build include it. To add only this extension to an existing installation:

```bash
python deploy/install.py --native-only --native-part ar_norm --cuda-home /usr/local/cuda
pip install --no-build-isolation --no-deps .
vllm-mach-install --apply
```

The installed package is `vllm-mach-ar-norm==0.1.0a1`, with native ABI `ar-norm-mxfp8-v1`. Missing or stale binaries fail explicitly for eligible models. There is no JIT build or experiment-directory lookup at startup.

`vllm-mach-serve` enables this producer for Dense by default. `--no-fused-ar-quant` disables only this producer; `--no-fused-ar-norm` also disables it. Direct Python users can set `VLLM_MACH_FUSED_AR_QUANT=auto|0|1` (default `auto`). Eligible calls require the prepared native MXFP6 MLP/GDN producers, the existing BF16/SM120/TP2/PP1 Dense manual AR path, hidden size 5120, `CompilationMode.NONE`, and a TRTLLM AllReduce workspace.

Only pure decode at physical M2/4/8/16/24/32 uses the new path. At Python dispatch/capture, metadata must describe a matching decode batch; CUDA Graph replay then retains that captured sequence, including any padded rows. M1, prefill, mixed or speculative batches, other shapes and unavailable workspaces retain the original method. MoE and compiled model paths remain unchanged. This does not impose a request-concurrency limit or reduce KV capacity; larger physical batches use the original kernels. The validated Dense model prepares 64 layers: 47 GDN input boundaries plus 64 post-attention boundaries. First-layer input and full-attention input normalization retain their existing paths.

BF16 normalized output remains available to GDN's BA projection. Codes/scales are passed explicitly to the QKV consumer after BA is launched, preserving overlap and stream ordering. No activation-address cache or new shared activation buffer is introduced. The collective uses vLLM's existing serialized-worker workspace contract.

## Numerical evidence

The installed kernel header is identical to the precision-tested implementation. Real layer inputs at M4/M32 and row slices for M1/2/8/16/24 passed byte comparisons for residuals, normalized BF16 values, FP8 codes, full packed scale buffers and W6A8 results. Changed-input CUDA Graph replay and poisoned scale padding are included. The original memory-sanitizer check reported zero kernel memory errors on both ranks.

The fixed four-domain suite has 256 queries and 10,479 target-token logprobs per physical-row setting. With FP32 SSM and the BF16 head, the fusion had zero differences from the prior exact-producer champion. Query-mean MAE against the archived, shape-matched BF16 reference was 0.0855531417 at M4 and 0.0913294934 at M32. These precision results are reused because the numerical kernel is unchanged; this is not a new BF16 run. It does not establish bitwise equivalence for arbitrary models or a new application-level accuracy score.

## Throughput evidence

The original A→B short run used Qwen3.8-27B MXFP6, dual RTX 5090, TP2, FP32 SSM, BF16 head/KV, 32 MLP replica layers and 8,218,214,400 KV bytes per rank. It submitted 64 requests with 3,000 input and 1,000 output tokens, at most 32 active, using `NONE` / `FULL_DECODE_ONLY`. After a complete same-shape warmup, two scored repeats yielded 1542.3023 tok/s without this fusion and 1552.1877 tok/s with it (+0.64095%). This is offline LLM end-to-end timing, not an HTTP or four-concurrency benchmark, and it is not a new NVFP4 comparison.

The installed-wheel B→A confirmation used the same contract and produced 1541.2975 tok/s without fusion and 1553.7055 tok/s with it (+0.80503%). Both corresponding scored-round differences were positive. Descriptive pooling across both orderings gives 1541.7997 → 1552.9462 tok/s (+0.72295%), four timed rounds per arm across two load pairs. This is not a confidence interval or a four-point throughput result. Scored windows had no new JIT messages; GPU process samples contained only the corresponding workers and both arms used 13,801 MHz memory clocks.

The target CUDA environment built and installed both wheels. The installed primitive passed all seven tested row sizes, including four changed-input graph replays. Related CPU tests passed 105 cases and skipped 8 CUDA-only cases; the installed primitive checks cover the newly added CUDA operation.

[Machine-readable aggregate results](data/ar-norm-mxfp8-20260920.json) contain the contract, both loading orders, scored times and reused precision evidence.

## Higher concurrency

A dedicated 48-request, 3k/1k compatibility run completed all 48,000 output tokens, with actual decode concurrency 48 for 909 steps per rank. It used a 10 GiB KV budget per rank, retained the same 32 replica layers, and exercised the original path above M32. It recorded two preemptions, so this is not a zero-preemption claim or a c48 throughput comparison. The diagnostic's extra zero-preemption assertion failed after generation; the completed-output and dispatch records are preserved.

With the original 8,218,214,400-byte KV budget, all 48 requests also completed, but peak actual decode concurrency was 45 with ten preemptions. That budget and all c32 measurements remain unchanged. Size KV for the intended workload; a larger request queue does not prove that all requests are active together.
