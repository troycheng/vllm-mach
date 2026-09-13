# ExLlamaV3 1.5.0 BF16 build

This profile applies Mach's downstream BF16 I/O interface to upstream `v1.5.0` (`0740edc2da569fb99174023c1d2988b1e98cb41e`). It retains the 1.5.0 multi-row GEMM implementation and adds the BF16 output arguments after `size_n_stride`. Native checks and TP2 service validation passed; throughput stayed within 0.14% of 1.4.9 in the [four-concurrency short regression](../../docs/exllamav3-1.5.0.md). Use the current source checkout for this profile; the published a10 source archive retains its 1.4.9 profile. The Mach a10 Python runtime needs no changes.

```bash
git clone --branch v1.5.0 --depth 1 https://github.com/turboderp-org/exllamav3.git
git -C exllamav3 apply --check /path/to/vllm-mach/profiles/exllamav3-1.5.0/bf16-io.patch
git -C exllamav3 apply /path/to/vllm-mach/profiles/exllamav3-1.5.0/bf16-io.patch
cd exllamav3
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=12.0a python -m pip install --no-deps --no-build-isolation .
```

Build in the vLLM 0.29 serving environment with PyTorch 2.13.0. Rebuild Mach's separate [M32](../../native/exl3_m32/README.md) and [Temporal M24](../../native/exl3_temporal_m24/README.md) extensions against this patched source. Existing vLLM/FlashInfer patches and the lossless-prefill extension are unchanged. Triton is already supplied by the vLLM environment; ExLlamaV3 1.5.0 imports it unconditionally.

For both extension builds, set `EXLLAMA_V3_SOURCE=/absolute/path/to/exllamav3/exllamav3/exllamav3_ext` to the patched 1.5.0 checkout above, rather than checking out the older revision in their standalone instructions. Keep the same PyTorch/CUDA environment and use `--no-deps --no-build-isolation`. This installation supplies Mach's native dependency; standalone ExLlamaV3 CLI dependencies are separate.

Mach retains its existing `hgemm` call for the reconstruction path. ExLlamaV3 1.5.0's new `hgemm_recon` uses FP16 partial sums with FP32 accumulation across slices on supported devices; this profile does not opt into that different arithmetic. MoE scheduling and ExLlamaV3-native GDN changes do not replace Mach's vLLM Dense/GDN paths.

The BF16 API remains a downstream integration, not an interface in the official ExLlamaV3 wheel. ExLlamaV3 is MIT-licensed; upstream source notices are retained.
