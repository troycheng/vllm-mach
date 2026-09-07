# Lossless BF16 prefill collective

Optional, version-locked TP2 SM120 implementation for BF16 M4096×H5120 AllReduce/residual/GemmaRMSNorm. It compresses only input peer reads, restoring BF16 bits before the existing FP32 sum. The sum-transfer leg and residual/norm arithmetic remain unchanged. Other shapes keep the existing vLLM dispatcher.

The codec uses 256-value blocks, fixed 512-byte slots, 384-byte compressed payloads and two-byte headers. Exponent ranges that cannot be represented exactly, zeros and special values select raw BF16 per block. This does not quantize activations or reduce the allocated workspace size.

## Build and enable

Use the complete [a6 runtime profile](../../profiles/vllm-0.28.0/README.md#checkpointtemporal-serving-configuration) with vLLM 0.28.0 and FlashInfer 0.6.18. This extension requires CUDA **13.0** and targets `sm_120f`, matching the tested reference compiler and fast-math settings. Other compiler versions are rejected by this source profile; they require new numerical validation.

```bash
cd native/lossless_prefill
CUDA_HOME=/path/to/cuda-13.0 MAX_JOBS=2 \
  python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/vllm_mach_lossless_prefill-*.whl
```

Install the matching Mach development wheel, then apply the thin vLLM caller patch before starting the service:

```bash
patch --batch --fuzz=0 -p1 -d /path/to/site-packages \
  < profiles/vllm-0.28.0/lossless-prefill.patch
export VLLM_MACH_LOSSLESS_PREFILL=1
```

The switch defaults to off. Unsupported shapes retain their existing dispatch. Enabling it without the native extension, on an unvalidated device, or with incompatible workspace metadata raises an error. The workspace must be the same `trtllm` object throughout the worker lifetime and provide at least 84,049,920 bytes per rank, including the header region. The pointer table is not the payload allocation. Changing the FlashInfer workspace ABI requires revalidation.

The native library is loaded with `torch.ops.load_library`, not imported as a Python extension. First use must happen outside CUDA Graph capture. Decode M24/M32 remains on its original one-shot path.

## Validation

The included boundary harness can run without model files:

```bash
python -c 'import importlib.util; print(importlib.util.find_spec("mach_lossless_prefill_ext").origin)'
GLOO_SOCKET_IFNAME=lo python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29581 \
  bench_ar_codec.py --library /path/printed/above.so \
  --capture-dir . --output-dir /path/to/new/results --smoke
```

It compares the installed FlashInfer implementation, a recompiled control and the packed path, including changing-input Graph replay. Real captured inputs can be supplied through `--capture-dir` with `--validate-only`; `bench_mixed_workspace.py` additionally covers mixed M24/M32 one-shot and M4096 two-shot use of a shared workspace. Those captured model inputs are not distributed.

For a diagnostic service run, also set `VLLM_MACH_LOSSLESS_PREFILL_VERIFY=1`. The first eligible call at each norm instance is compared bitwise with installed FlashInfer, checking both residual and normalized outputs. This adds communication and synchronization; **disable it for performance measurement**. Logs report activation and the number of verified norm instances per rank.

The source experiment recorded a +0.88% pooled c32 service result over four A/B/B/A lifecycles. That is historical evidence for the original runtime, not a Mach release speedup claim. “Lossless” describes only this communication replacement, not the quantized model's overall accuracy. See the [Mach integration record](../../docs/lossless-prefill.md) for current port acceptance.
