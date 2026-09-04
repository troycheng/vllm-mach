# vLLM 0.28 runtime profile

The Dense MXFP6 kernel can register through the vLLM plugin interface. Stream-K
workspace planning needs two additional lifecycle calls that vLLM 0.28 does not
expose as plugin hooks: one before CUDA Graph capture and one on the graph-capture
stream.

Apply `mxfp6-graph-warmup.patch` to the exact vLLM 0.28.0 source tree before
building the serving image. The patch adds only those lifecycle calls; kernel
selection, checkpoint mapping, and the CUDA implementation remain outside vLLM.

```bash
git -C /path/to/vllm apply \
  /path/to/vllm-mach/profiles/vllm-0.28.0/mxfp6-graph-warmup.patch
```

Do not apply this profile to another vLLM version without revalidating its patch,
imports, model load, changing-input graph capture, and inference behavior.
