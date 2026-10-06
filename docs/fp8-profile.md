# Qwen3.5-4B block-FP8 profile

`qwen35-4b-block-fp8-v1` is an explicit, independent profile for the existing Qwen3.5-4B compressed-tensors block-FP8 checkpoint on one RTX 5090 (SM120). It provides four separately selectable optimizations: n64 GEMM, ordered low-M GEMM, SiLU/block quantization and mixed FA2. It uses BF16 activations, KV cache and the ordinary model head, with FP32 SSM state.

The [October 6 qualification](fp8-qualification.md) records byte-identical fixed-gold M4/M32/M64 results and one stock/all serving pair at c4/c16/c32/c64, with measured throughput changes of +9.03%/+6.18%/+5.97%/+6.99%. All 680 requests per arm succeed. The final source build has verified runtime byte identity to the tested image.

The profile keeps the checkpoint's block-FP8 weights and FP32 block scales. MXFP8 E8M0 scales have a different format and cannot use this entry. The [MXFP8 champion profile](mxfp8-champion.md) and native MXFP6 profiles have their own installers and launchers.

## Build

Build from the checkout containing `deploy/Dockerfile.fp8`. The existing `v0.1.1` MXFP6 release instructions do not install this profile.

```bash
docker buildx build --load \
  --build-arg MAX_JOBS=2 \
  -f deploy/Dockerfile.fp8 -t vllm-mach-fp8:local .
```

The Dockerfile pins its vLLM base image by digest, compiles the two native extension wheels for SM120, installs Mach without replacing the base packages, and applies the block-FP8 source profile. Native build parallelism defaults to two jobs. The installer records its result at `/opt/mach/runtime-source-install.json`.

The package contract is checked at startup:

| Package | Required version |
|---|---|
| vLLM | 0.29.0 |
| Torch | 2.13.0; the pinned image provides the CUDA 13.0 build |
| Triton | 3.7.1 |
| FlashInfer Python and cubin packages | 0.6.18 each |
| CUTLASS Python DSL | 4.6.2 |
| CUDA Python bindings | 13.3.1 |

The native build uses NVCC 13.0.88 and FlashInfer's pinned CUTLASS headers. The [n64 wheel](../native/block_fp8/README.md) records compiler, source and header identities in `build_metadata.json`. The [SiLU wheel](../native/fp8_activation/README.md) is independent of CUTLASS and MXFP6. Both are built before serving, rather than compiled by an individual request.

For an existing compatible installation, install both prebuilt native wheels and the Mach package, then inspect and apply the source profile:

```bash
vllm-mach-fp8-install
vllm-mach-fp8-install --apply
```

The first command stages and checks the patch without changing installed files. Application validates package versions and each source fingerprint, compiles staged Python source, and uses the shared rollback mechanism. Repeating an already applied installation does not wrap or patch the runtime again. A different vLLM file or package version requires a reviewed profile update; installing a similarly named version is insufficient.

## Start

Mount a compatible local block-FP8 checkpoint and its tokenizer as read-only directories. The profile does not download, reconstruct or requantize a model.

```bash
mkdir -p ./fp8-run

docker run --rm --gpus '"device=0"' --ipc=host \
  -p 8000:8000 \
  -v /absolute/path/to/block-fp8-checkpoint:/model:ro \
  -v /absolute/path/to/tokenizer:/tokenizer:ro \
  -v "$PWD/fp8-run:/run" \
  vllm-mach-fp8:local /model --tokenizer /tokenizer \
  --run-dir /run --host 0.0.0.0 --port 8000 \
  --features n64 ordered silu fa2
```

The checkpoint must have Qwen3.5-4B text geometry: hidden size 2560, intermediate size 9216, 32 layers, 16 attention heads, 4 KV heads and head dimension 256. A checkpoint with a usable tokenizer in the same directory can omit `--tokenizer` and the separate tokenizer mount.

Outside the image, the equivalent launcher is:

```bash
vllm-mach-fp8-serve /absolute/path/to/block-fp8-checkpoint \
  --tokenizer /absolute/path/to/tokenizer --run-dir ./fp8-run \
  --features n64 ordered silu fa2
```

The launcher selects TP1, BF16, compressed-tensors quantization, the CUTLASS linear backend, `FLASH_ATTN`, text-only execution and Model Runner V2. It fixes maximum model length 8192, maximum sequences 128, scheduler token budget 2048, 19 GiB KV memory, FP32 SSM cache and no prefix caching. It creates fresh vLLM, Inductor and Triton cache directories for each startup. `--print-command` shows the generated command and selected features without launching the server.

Startup checks the package/source profile and model geometry before enabling the operators. Worker registration runs after device initialization and before model compilation. Launch settings and worker qualification receipts are written in `--run-dir`. Linear and SiLU receipts must show actual execution of selected routes; registration alone is not proof of use. Mixed FA2 needs an actual mixed step, so pure-decode warmup cannot establish its coverage.

## Select the four features

The default selects all four. `--features` selects a nonempty subset and resets the unselected feature flags to zero; an existing shell flag cannot silently add an optimization.

| Feature | Environment flag set by the launcher | Eligible work |
|---|---|---|
| `n64` | `VLLM_MACH_FP8_N64` | N2560, K4096/9216 block-FP8 GEMM at M16–128 divisible by eight; exact N128-to-N64 scale duplication |
| `ordered` | `VLLM_MACH_FP8_ORDERED` | The same N/K geometry at M1–8; raw partials reduced in the selected ordered accumulation path |
| `silu` | `VLLM_MACH_FP8_SILU` | The selected BF16 SiLU/multiply and E4M3 group128 quantization entry; original mutation and fallback contract |
| `fa2` | `VLLM_MACH_FP8_FA2` | A one-token decode prefix plus one or two longer prefills in the selected BF16 FA2 geometry |

For example, `--features n64 ordered` enables only the linear routes. `--features fa2` enables only the attention backport. Feature flags accept `0` or `1` and cannot change after worker registration. See [linear](fp8-linear.md), [SiLU quantization](fp8-activation.md) and [attention](fp8-attention.md) for their detailed contracts.

Unsupported operator shapes take the stock operator path. Unsupported profile configurations fail at startup: other model geometry or GPU architecture, TP/PP greater than one, DCP, DBO, speculative decoding, quantized KV, non-FP32 SSM or CPU weight offload. Online weight reload and weight transfer are rejected; restart the worker to replace weights so derived scales and captured graphs are rebuilt.

The attention optimization preserves the framework's graph selection and request ordering. CUDA graph capture and unsupported masks, windows, sinks, dtypes or page geometry use the original FA call. There is no separate local attention graph policy.

## Precision and qualification

These optimizations do not introduce a new quantization format or change the checkpoint. n64 duplicates existing FP32 weight scales exactly; ordered GEMM and SiLU preserve their selected arithmetic contracts; mixed FA2 invokes the existing attention implementation on request slices. Component tests and matching fixed-gold model records validate this source port's tested boundary.

Package versions, native builds and the compilation recipe belong to the precision contract. In particular, an unchanged RMSNorm source can select different reduction configurations through autotuning and produce different model records. Cold-start cache isolation alone does not fix that choice.

The launcher selects the packaged `production` recipe through `VLLM_MACH_FP8_COMPILE_MODE` and sets `TORCHINDUCTOR_COMPILE_THREADS=1`. The recipe matches six stock RMS reduction sites by source, shape and compiler descriptors, then fixes their selected launch configuration. It rebuilds them from source; it does not consume a frozen binary cache. Unrelated kernels retain their normal heuristics. A recognized RMS boundary that changes fails rather than silently selecting another recipe.

The worker installs the recipe before generated modules are compiled and verifies binding coverage and actual launch configurations after warmup. Its receipt records the policy mode and hash, required/missing bindings and observed configurations. These are compilation/configuration records, not GPU execution counts. The separate `quality4`, `quality32` and `quality64` recipes belong to the matching precision-validation drivers; the serving launcher uses `production`. Model records still must pass against the matching stock arm after a recipe change.

Each final fixed-M check covers 256 queries and 10,479 gold logprobs, with MAE zero and raw record bytes equal to its stock reference. The initial M4 RMS autotune mismatch was repaired by the exact recipes above. The [qualification report](fp8-qualification.md) separates these private-corpus precision checks from the public 3k/1k serving protocol, reports mean/p99 TTFT as well as throughput, and makes no statistical significance or individual feature-gain claim. The client summaries' `complete:false` means four selected points of a six-point protocol; all selected points succeed.

The final image is `sha256:f1423984dfd1d79d1de4bcb72a6ec7bc7e51ea3c29d8d1a5cc218d66a93fc55e`. All 85 compared installed runtime files, including both native libraries, are byte-identical to tested image `sha256:9eabff48de8a713589928824aef0b353dd92cab4e8e5fa05810b804441fe33e9`; only the native build-metadata source hash changes after whitespace cleanup. Checkpoint file hashes in the receipt are previously frozen identities reused through the same read-only model mount, with no inferred download source or new full-checkpoint hash pass.

The mixed FA2 algorithm is attributed to [LiRunGuo's vLLM PR #58013](https://github.com/vllm-project/vllm/pull/58013). Mach's contribution is its narrow backport and qualification. The SiLU entry is related to [vLLM PR #45055](https://github.com/vllm-project/vllm/pull/45055), while this profile retains its separately qualified scalar CUDA implementation. Upstream acceptance and Mach release qualification are separate milestones.
