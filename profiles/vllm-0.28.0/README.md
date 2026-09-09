# vLLM 0.28 runtime profile

The Dense MXFP6 kernel can register through the vLLM plugin interface. Stream-K
workspace planning needs two additional lifecycle calls that vLLM 0.28 does not
expose as plugin hooks: one before CUDA Graph capture and one on the graph-capture
stream.

Apply `mxfp6-graph-warmup.patch` to the exact vLLM 0.28.0 source tree before building the serving image. The patch adds only those lifecycle calls; kernel selection, checkpoint mapping, and the CUDA implementation remain outside vLLM.

```bash
git -C /path/to/vllm apply \
  /path/to/vllm-mach/profiles/vllm-0.28.0/mxfp6-graph-warmup.patch
```

The optional Qwen3.8-27B TP2 fused collective also requires `qwen38-fused-ar-rmsnorm-mxfp8.patch` in vLLM and `flashinfer-mxfp8-packed-layout.patch` in the installed FlashInfer package tree. Apply both before FlashInfer JIT compilation:

```bash
git -C /path/to/vllm apply \
  /path/to/vllm-mach/profiles/vllm-0.28.0/qwen38-fused-ar-rmsnorm-mxfp8.patch

patch --batch --fuzz=0 -p1 -d /path/to/site-packages \
  < /path/to/vllm-mach/profiles/vllm-0.28.0/flashinfer-mxfp8-packed-layout.patch
```

For `flashinfer-python==0.6.16.post3`, also apply the SM120 cluster-size guard:

```bash
patch --batch --fuzz=0 -p1 -d /path/to/site-packages \
  < /path/to/vllm-mach/profiles/vllm-0.28.0/flashinfer-sm120-cluster-limit.patch
```

FlashInfer `0.6.18` already contains the cluster-size guard. The accepted patched header SHA256 values are `8e3f0d82c307da6d0b7be769cb672164c14bd8594eb5dc8dbad8fb2091b331df` (the earlier profile, including patched `0.6.16.post3`) and `049e8b8c0b9f866d1a49247399a17de0809f3779521753debcd648b7888b1a4e` (`0.6.18` with kernel-side PDL guards). Package version alone does not establish compatibility.

The communication module must be compiled from this patched source. FlashInfer prefers the prebuilt `trtllm_comm.so` in `flashinfer-jit-cache`, even when the installed header has changed. The tested cache wheel used the unpatched scale layout and produced incorrect GEMM outputs. A header check cannot validate a prebuilt binary, so Mach rejects that build path as well as unknown header hashes.

In the environment dedicated to this profile, remove the prebuilt cache package and restart any processes that imported FlashInfer:

```bash
python -m pip uninstall flashinfer-jit-cache
```

Keep `flashinfer-python` and `flashinfer-cubin` installed. A complete CUDA development toolkit is required; the tested build used CUDA 13.2. Uninstalling the cache package also makes its other prebuilt modules unavailable, so do this in the serving environment, not a shared environment used by unrelated services. After applying the patches, run the combined source/build-path check before starting vLLM:

```bash
python -c 'from vllm_mach.exl3.fused_allreduce import verify_flashinfer_profile; verify_flashinfer_profile()'
```

## Sampling metadata

`sampling-device-metadata.patch` is independent of the MXFP6 patches. Apply it to the exact vLLM `0.28.0` source tree:

```bash
git -C /path/to/vllm apply \
  /path/to/vllm-mach/profiles/vllm-0.28.0/sampling-device-metadata.patch
export VLLM_MACH_SAMPLING_DEVICE_METADATA=1
```

Set the variable before starting vLLM. The patch copies temperature and seed tensors to device storage before the Gumbel sampler reads them. It leaves logits and sampling math unchanged. The default is disabled; it also copies already-device-resident metadata when enabled, so it is intended for the tested mapped-host-metadata path, not as a universal sampling optimization. See [validation limits](../../docs/experimental-decode.md).

Do not apply this profile to another vLLM version without revalidating its patch,
imports, model load, changing-input graph capture, and inference behavior.

## Checkpoint/Temporal serving configuration

`qwen38-runtime-alignment.patch` adds the reference SM120 TP2 AllReduce size limits and backports fused QK norm/MRoPE handling from [vLLM #52676](https://github.com/vllm-project/vllm/pull/52676). It does not include the reference image's experimental MoE runner. Apply it alongside the patches above:

```bash
patch --batch --fuzz=0 -p1 -d /path/to/site-packages \
  < profiles/vllm-0.28.0/qwen38-runtime-alignment.patch
```

Install the optional [FlashInfer GDN profile](../flashinfer-0.6.18-gdn/README.md), B12X 1.3.0, and the Temporal native extension, then source `qwen38-checkpoint-temporal.env`. Set `VLLM_MACH_MXFP6_CHECKPOINT` explicitly to the paired original MXFP6 checkpoint. The environment file enables online K6 embedding, B12X/fused reconstruction, the `trtllm` collective backend and GDN. Online embedding changes the stored numerical representation; these settings are opt-in and are not the package defaults.

Use TP2, BF16 KV, TRITON_ATTN, max-model-len8192, max-num-seqs32, max-num-batched-tokens4096, chunked prefill, no prefix caching, text-only input, and FULL_DECODE_ONLY graph sizes1/2/4/8/16/24/32 for the reference configuration. Do not substitute a whole-model MXFP6 quantization configuration for the EXL3 provider.

The profile files ship in the a6 source archive, not the Python wheel. Use a tagged checkout so the installer, overlay and environment file match:

```bash
git clone --branch v0.1.0a6 --depth 1 https://github.com/troycheng/vllm-mach.git
cd vllm-mach
```

Prepare a dedicated environment with vLLM 0.28.0, the Mach wheel, patched [ExLlamaV3 1.4.8](../exllamav3-1.4.8/README.md), `mxfp6-sm120==0.2.1`, `flashinfer-python==0.6.18`, and `b12x==1.3.0`. Apply the graph-warmup, fused collective, sampling, FlashInfer layout and runtime-alignment patches above before running the GDN installer. Build the [Temporal extension](../../native/exl3_temporal_m24/README.md) in that environment. Do not reuse an unpatched FlashInfer communication binary.

After installing those components:

```bash
python profiles/flashinfer-0.6.18-gdn/install.py --apply
source profiles/vllm-0.28.0/qwen38-checkpoint-temporal.env
export VLLM_MACH_MXFP6_CHECKPOINT=/path/to/paired-original-MXFP6-checkpoint
python -c 'from vllm_mach.exl3.fused_allreduce import verify_flashinfer_profile; verify_flashinfer_profile()'

vllm serve /path/to/EXL3-checkpoint \
  --quantization exl3 --tensor-parallel-size 2 \
  --dtype bfloat16 --kv-cache-dtype auto \
  --gpu-memory-utilization 0.9 --max-model-len 8192 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

Check both worker logs for `cuda_sm120_persistent`, both Temporal bundle shapes, fused FlashInfer collective activation and completed Graph capture. A healthy HTTP endpoint alone does not establish that these optional paths are active. Run a representative full-concurrency warmup before steady-state measurement; the first c16 round showed extra latency in acceptance. See the [combined integration result](../../docs/champion-alignment.md).

## Extended prefill profile

The v0.1.0a8 source extends the checkpoint profile with direct SUM, 32 observed prefill shapes, and M32 BA overlap. Install the matching v0.1.0a8 wheel, build [lossless native package 0.1.0a4](../../native/lossless_prefill/README.md), and update the GDN overlay before starting workers. On top of the complete profile above, apply these caller patches in order:

```bash
patch --batch --fuzz=0 -p1 -d /path/to/site-packages < profiles/vllm-0.28.0/lossless-prefill.patch
patch --batch --fuzz=0 -p1 -d /path/to/site-packages < profiles/vllm-0.28.0/long-prefill.patch
source profiles/vllm-0.28.0/qwen38-checkpoint-long.env
```

If the M4096 patch is already installed, apply only the second patch. The environment file keeps the same checkpoint/Temporal settings and explicitly enables the prefill and BA switches; both diagnostic verification switches are set to zero. Base package defaults remain off. The latest source, wheel and overlay must be used together; these additions are not in the a7 source archive. See [integration evidence and scope](../../docs/long-prefill.md).
