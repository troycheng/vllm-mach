# vLLM 0.29 runtime port

Development profile against `v0.29.0` (`98dff2a81d747d1dba01a47f939f48c3526d4206`), for Mach `0.1.0a10.dev0`. It carries the a9 checkpoint-hybrid paths onto the new runtime. The published a9 release remains on vLLM 0.28.

TP2 service validation is complete: 40/40 task checks and 2,592 BA state/output comparisons passed. The same-device 3k/1k short regression retained throughput at c4/c16/c24/c32. See [upgrade results and test setup](../../docs/dependency-upgrade.md).

`mxfp6-graph-warmup.patch` ports the two MXFP6 workspace lifecycle calls: before memory profiling and on the capture stream. MRV2 can capture profiling graphs before the regular kernel warmup, so workspace planning now runs at the start of memory sizing. vLLM 0.29 already passes the current stream to `torch.cuda.graph`, so that part of the old patch is omitted. The patch preserves 0.29's graph-memory profiling code.

`runtime.patch` contains the Qwen fused AllReduce/RMSNorm/MXFP8 caller, SM120 TP2 size limits, device sampling metadata, and direct/long lossless prefill dispatch. The QK norm/MRoPE backport is already in 0.29 and is omitted.

The GDN installer selects the matching 0.29 source overlay. It preserves upstream CPU handling, prefill sequence-length conversion, sigmoid/speculative decode support and missing-metadata warmup behavior. Mach's FP16 state support remains restricted to its existing non-speculative SiLU profile. BA overlap remains available at M32 and, with FP16 state enabled, M16/M24.

Use the official 0.29 dependency set: PyTorch 2.13.0, FlashInfer Python/cubin 0.6.18 and CUTLASS DSL 4.6.2. Retain B12X 1.3.0 and MXFP6 SM120 0.2.1. Keep the FlashInfer packed-scale layout and binary-source checks; upgrading a package does not replace those requirements.

`flashinfer-local-ipc.patch` carries the validated image's local TRT-LLM IPC allocation fix. This path does not use RDMA and must not request RDMA-capable memory on SM120. The patch adds an allocation option to `SymmDeviceMemory`, defaults it to the upstream behavior, and disables it only in the local TRT-LLM caller. MNNVL callers retain their original allocation flags.

## Installation

Use a dedicated environment. Install the development Mach wheel, [patched ExLlamaV3 1.4.9](../exllamav3-1.4.9/README.md), and the [M32](../../native/exl3_m32/README.md) and [Temporal](../../native/exl3_temporal_m24/README.md) extensions built against that source. The unchanged [lossless prefill extension](../../native/lossless_prefill/README.md) still uses its CUDA 13.0 build contract.

From the matching Mach checkout, apply these patches to the environment's site-packages directory before importing vLLM or FlashInfer:

```bash
site=$(python -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file(""))')
for p in mxfp6-graph-warmup runtime flashinfer-local-ipc; do
  patch --batch --fuzz=0 -p1 -d "$site" < "profiles/vllm-0.29.0/$p.patch"
done
patch --batch --fuzz=0 -p1 -d "$site" < profiles/vllm-0.28.0/flashinfer-mxfp8-packed-layout.patch
python -m pip uninstall -y flashinfer-jit-cache
python profiles/flashinfer-0.6.18-gdn/install.py --apply
python -c 'from vllm_mach.exl3.fused_allreduce import verify_flashinfer_profile; verify_flashinfer_profile()'
```

The official 0.29 image does not ship the ten legacy experimental FlashInfer GDN files used by this profile. The installer adds those files and checks both upstream vLLM file hashes before writing. It rejects unrecognized existing files and is idempotent after installation. Do not apply the old 0.28 runtime patches as well.

For the a9 FP16-state checkpoint-hybrid configuration:

```bash
source profiles/vllm-0.29.0/qwen38-checkpoint-fp16-ssm.env
export VLLM_MACH_MXFP6_CHECKPOINT=/path/to/paired-original-MXFP6-checkpoint
```

Retain the [a9 serving arguments](../../docs/fp16-ssm.md), including TP2, BF16 activations/KV, `--mamba-ssm-cache-dtype float16`, non-speculative decoding, and graph sizes 1/2/4/8/16/24/32. The environment file keeps standalone FlashInfer AllReduce disabled, as in the previous profile. Mach's fused FlashInfer collectives remain enabled; this avoids the new default standalone path allocating their shared workspace first.
