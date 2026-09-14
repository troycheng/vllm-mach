# Owner prefill

Optional TP2 owner-residual path for the Qwen3.8-27B checkpoint profile. It covers physical prefill batches of 3000–4096 rows at hidden size 5120; other shapes retain the existing runtime. The cooperative CUDA kernels require SM120 with 170 SMs. This is a distinct prefill path: rank64 GU and Temporal decode remain unchanged.

Each rank owns a contiguous row range between norms. QKV/QKVZ inputs use gathered MXFP8 values and 128-row scale atoms; GDN BA uses the original loaded BF16 parameters and the original physical matrix shape. Ragged transfers pad only communication packets, never attention inputs. The final norm gathers full BF16 output for the LM head.

The default 32 even-numbered layers replicate the peer's original packed MXFP6 gate/up and down weights. Both original TP branches run locally on owner rows, preserving their independent BF16 rounding before the original rank-ordered sum and norm. Replicas require **3,342,336,000 bytes (3.113 GiB) per rank**, plus an independent collective workspace and BA replicas. No KV setting is changed automatically.

## Build

Install the vLLM 0.29.0 / ExLlamaV3 1.5.0 source profile first. Build against the same PyTorch and FlashInfer 0.6.18 installation used for serving. The selected implementation was built with CUDA **13.2**, C++17 and `sm_120f`; preserve these compiler and fast-math settings when reproducing its numerical checks.

```bash
cd native/owner_prefill
CUDA_HOME=/usr/local/cuda-13.2 MAX_JOBS=2 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/vllm_mach_owner_prefill-*.whl
cd ../..
python profiles/vllm-0.29.0/install-owner-prefill.py
export VLLM_MACH_OWNER_PREFILL=1
```

Apply the four model hooks after `profiles/vllm-0.29.0/runtime.patch`. The installer checks the vLLM version and exact patch context and is safe to run again. Install the matching Mach Python source as well; do not apply an old full-model overlay.

`VLLM_MACH_OWNER_MLP_LAYERS` optionally accepts a JSON list of distinct indices; the default is `[0,2,...,62]`. Use `[]` for owner communication without MLP replicas. Both modes require a worker restart. Owner prefill is disabled by default.

For numerical acceptance, set `VLLM_MACH_OWNER_VERIFY=2` on a diagnostic service. The first two real eligible prefill forwards compare residuals, norms, MXFP8 values/scales, QKV/QKVZ, BA, both local MLP branches and final gathered output with the original operations. Failure aborts the request, and detailed results remain in `model._owner_prefill_state.checks`. Set `VLLM_MACH_OWNER_VERIFY_ONLY_RAGGED=1` to reserve these checks for non-4096 rows. Disable verification for throughput measurement.

The wheel contains four compiled extensions and checks ABI `owner-prefill-v1`; it does not load an experimental `.so` or compile at service startup. All local `.cu`/`.cuh` dependencies are included in the source distribution. FlashInfer, CUTLASS and spdlog headers come from the pinned FlashInfer package. The retained fused collective header derives from FlashInfer's Apache-2.0 implementation; the owner and transport kernels preserve its residual/norm arithmetic, barriers and programmatic launch dependency protocol.
