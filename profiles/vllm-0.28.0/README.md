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

FlashInfer `0.6.18` already contains the cluster-size guard, but its version number does not establish fused-path compatibility. The plugin requires the exact patched header SHA256 `8e3f0d82c307da6d0b7be769cb672164c14bd8594eb5dc8dbad8fb2091b331df`, including patched `0.6.16.post3`. The previously documented claim that all patched `0.6.18` inputs produce this hash was too broad.

A tested `0.6.18` wheel instead produced header `049e8b8c0b9f866d1a49247399a17de0809f3779521753debcd648b7888b1a4e`, with additional kernel-side PDL guards. A local acceptance experiment allowing that header failed the TP2 task regression. It remains rejected; do not bypass the guard. See [validation.md](../../docs/validation.md) for the tested configuration. After applying the patches, check the installed header before allocating GPUs or starting a service:

```bash
python -c 'import flashinfer; from vllm_mach.exl3.fused_allreduce import _verify_flashinfer_header; _verify_flashinfer_header(flashinfer)'
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
