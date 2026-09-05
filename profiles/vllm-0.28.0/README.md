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

FlashInfer `0.6.18` already contains that cluster-size guard and only needs the packed-layout patch. Both supported inputs produce the same final header SHA256: `8e3f0d82c307da6d0b7be769cb672164c14bd8594eb5dc8dbad8fb2091b331df`. The plugin verifies this hash when `VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8=1` is selected.

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
