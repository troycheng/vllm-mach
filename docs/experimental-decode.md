# Experimental decode paths

These options are available in `0.1.0a3`. They are disabled by default and require `EXL3_BF16_IO=1`, the BF16 API described in the main README, and compatible grouped QKV/QKVZ bundles.

| Option | Dispatch |
|---|---|
| `EXL3_BF16_IO_M24=1` | Exact M24: M16 + M8 |
| `EXL3_BF16_IO_M32=1` | Exact M32: M16 + M16 |
| `EXL3_BF16_IO_TILE_M32=1` | Use the optional native M32 module when M32 dispatch is enabled |

Here M is the number of input rows in an operator call, not the configured concurrency. Other row counts retain their existing dispatch. These options do not change MXFP6 projection routing, prefill, or `lm_head`.

```bash
export EXL3_BF16_IO=1
export EXL3_BF16_IO_M24=1
export EXL3_BF16_IO_M32=1
# Requires the separately built module:
export EXL3_BF16_IO_TILE_M32=1
```

The M32 module has [separate build instructions](../native/exl3_m32/README.md). If it is absent, M32 falls back to M16 + M16 with a warning. An installed module with an incompatible API fails explicitly; loader or binary errors are not silently hidden. CUDA Graph priming uses the same row selection as execution. Each captured M32 call owns its lock workspace, which the kernel resets on the current stream.

The optional [sampling metadata patch](../profiles/vllm-0.28.0/README.md#sampling-metadata) stages temperature and seed tensors on the GPU before vocabulary tiles read them. It is independent of the EXL3 quantization path and remains opt-in.

## Evidence and limits

The originating experiment used Qwen3.8-27B K5/K6, vLLM 0.28.0, TP2 on two RTX 5090 GPUs. A repeated 1K-input/256-output diagnostic measured the combined true-M32 and sampling-metadata candidate at +4.576% throughput for c32 and +1.495% geometric mean across c4/c16/c24/c32. The control already included the M24/M32 split paths. These numbers do not isolate either change and do not establish a gain on a 3K/1K production workload.

M24 split execution was not bitwise-identical to the earlier full-model reference: the historical 48-case check reported mean KL 0.002073, p99 KL 0.023102, and 48/48 top-1 agreement. It must not be described as a lossless full-model change.

On 2026-09-05, the Mach port passed 60 CPU tests with 3 skipped. A bounded GPU check used real checkpoint slices from both TP ranks, one GDN QKVZ layer and one full-attention QKV layer. M24 split, M32 split, and true M32 matched the explicit M16-chunk reference bitwise in eager execution and CUDA Graph replay for three changing inputs each. These selected projections were K6; this check does not establish K5 kernel coverage or full-model fidelity. The sampling patch produced identical sampled IDs with changing logits, temperatures, and seeds at row counts 4, 16, 24, and 32.

The M32 extension was also rebuilt using the pinned public ExLlamaV3 headers, Python 3.12, PyTorch 2.13.0+cu130, and CUDA 13.2, targeting SM120a. That build passed the same GPU checks. Its shared-library SHA256 was `ed0aacdbc79496d32e4f2bf49249e837f4e23fcb6de45c051cf4cc3a8ab86402`. This verifies the documented source build for that environment, not binary compatibility with other runtimes.

The subsequent Mach TP2 service check enabled these options together with the EXL3/MXFP6 profile and fused FlashInfer AllReduce/RMSNorm/MXFP8 path. It captured all seven graph sizes, became healthy, and completed the existing 40-task retention suite at a maximum concurrency of 32. All 40 tasks passed with no pass/fail regressions against the stored reference; 33 outputs matched exactly. Both worker processes loaded the rebuilt M32 library.

The task suite is a functional regression check, not a full-distribution fidelity test or a throughput benchmark. The service check used an 8,192-token model limit; it does not extend validation to longer contexts.
