# Install the MXFP8 runtime source profile

The `qwen35-4b-mxfp8-champion-v1` profile installs a reviewed source delta on the public vLLM 0.29.0 runtime. It does not require the original experimental image. Install the Mach wheel and its native projection/head dependencies separately, then run this installer before starting workers:

```sh
python -m vllm_mach.mxfp8.install
python -m vllm_mach.mxfp8.install --apply
```

The first command is a dry-run. `--site /path/to/site-packages` selects a directory containing `vllm/`; package-version checks still apply. The direct Python API is `install_profile(site, apply=False)`. Offline source audits can pass `verify_packages=False`; the CLI has no such bypass. `inspect_sources(site)` verifies an installed source tree without importing vLLM, Torch, Triton, or CUDA.

`data/runtime_sources.json` records upstream, resulting and frozen comparison SHA256 values, the complete runtime patch checksum, package versions, file purposes, and the formalization boundary. It contains no model references, old workspace paths, or copied experiment receipts. Required runtime packages are vLLM 0.29.0, Torch 2.13.0, Triton 3.7.1, FlashInfer Python/cubin 0.6.18, CUTLASS DSL 4.6.2, and CUDA bindings 13.3.1. Native MXFP8/dual/BA and NVFP4 head dependencies remain the complete profile's responsibility.

The installer accepts the pinned public source tree or its own current resulting profile. Public sources receive the complete `runtime.patch`; an already installed current profile is left unchanged. It rejects older or unknown source bytes, a partial installation, symlinks, incompatible versions, and a corrupt bundled patch. Every hunk applies at its stated offset with exact context; offsets and fuzz are never accepted. All resulting hashes and Python syntax are checked in staging before any source is replaced. Applying an already installed profile makes no changes.

Apply to a stopped runtime. Publication takes an exclusive installer lock, rechecks the staged input bytes, replaces each whole file atomically on its filesystem, and restores already replaced files if a later publication fails. Permissions are preserved. This provides rollback for reported installation failures, not a filesystem transaction across process crashes or power loss. The installer builds no MoE, allreduce, IPC, CUDA extension, or model library.

## Source boundary

The public base plus Mach's existing MXFP6 common patch was compared with the frozen runtime during initial formalization. It was insufficient: the GDN and FlashInfer changes are larger than import substitutions. With the subsequent [shared GDN optimizations](gdn-runtime.md), the independent runtime patch changes eleven files and pins twenty runtime sources:

| Source | Result |
| --- | --- |
| `envs.py` | Frozen defaults, flags, and compilation-factor definitions |
| `utils/flashinfer.py` | Frozen XQA eligibility, autotuning, and quantization helpers |
| `qwen_gdn_linear_attn.py` | Existing SM120 prefill/FLA gates and custom-op boundary, plus guarded indexed BF16 state IO, strided mixed decode and normalization layout reuse |
| `v1/attention/backends/gdn_attn.py`, `utils.py` | Shared FlashInfer offsets, backend-specific metadata and packaged aligned state-index kernel |
| `mamba/ops/gather_initial_states.py` | Zero only uninitialized selected state pages |
| `flash_linear_attention/ops/fused_sigmoid_gating.py` | Caller-owned output buffer and supported strided gate inputs |
| `v1/attention/backends/flashinfer.py` | Complete frozen source, byte-for-byte; separate K/V head storage, FP8 KV update, XQA metadata and decode |
| `v1/worker/gpu/cudagraph_utils.py` | Frozen capture candidates/order and capture stream; package import for the empty MXFP6 warmup |
| `v1/worker/gpu/model_runner.py` | Public full-vocabulary runner with the frozen prompt-logprob profiling warmup |
| `v1/worker/gpu_worker.py` | Frozen graph-memory/lifecycle behavior; removes verified disabled legacy MXFP6 adapter calls |
| Public logits processor, Qwen3.5 model, sampler/states/penalties/input batch, prompt-logprobs, FLA recurrence and MXFP8 FlashInfer linear kernel | Exact public source guards |

The installer omits the legacy model runner, MoE modifications, and both native allreduce source changes. The complete 4B MXFP8 launcher still requires its model-specific V2/TP1 contract with AR fusion disabled. The shared source overlay also serves the 2B MXFP8 profile and the qualified 2B NVFP4 path; this does not broaden either launcher's model admission. It does not install the separate MXFP6 profile's MoE/AR/IPC additions.

## Arithmetic and operative changes

FlashInfer attention source bytes retain the frozen implementation. GDN now includes the separately qualified copy, metadata and state-index optimizations described in [Shared GDN/Mamba runtime](gdn-runtime.md). Indexed SM120 BF16 prefill requires support in the installed FlashInfer build and otherwise falls back. CUDA graph candidate selection and capture streams remain frozen. The prompt-logprob profiling helper and its launch are retained, including the ordinary prompt-logprobs serving entrypoint used for quality checks.

The public logits processor's `forward`, `_apply_head`, `_get_logits` and `_gather_logits`, and the public sampler's `apply_sampling_params` and `sample`, have identical ASTs to their frozen counterparts. The formal NVFP4 full-vocabulary head replaces `_apply_head` after compile; it does not need the frozen compact logits helpers. Compact hybrid modes and vocab-parallel greedy are disabled. The fixed sixpoint requests carry explicit seeds; the frozen sampler's explicit-seed guard rejects its compact alternatives, so it reaches the normal full-logits sampler. The public runner's normal complete-vocabulary branch is retained, rather than importing unavailable compact sampling helpers. This is an operative source simplification, not an assertion that arbitrary compact/unseeded workloads are supported.

The public `vllm.v1.worker.gpu.warmup` source has SHA256 `ca8e7bdcc0edd429904c8c509f7bdb5327e02824eebd24cb8886460568d75682`; the frozen source has SHA256 `def9c07eb673b2fd479edf4ddf48483c267563f6eb8dfa5cb6a6cc608bdb6f1e`. The frozen version appends three real request/sample/cleanup warmups for greedy, full-distribution and top-k sampling, gated by model helper methods. The public version omits that block and its Qwen model omits those helpers. The accepted startup log confirms all three compact fast paths ran despite the disabled hybrid/greedy flags: these warmup requests are unseeded and request no logprobs. Removing them is an operative startup change, not a no-op or a claim that all warmup launches are equivalent. These hashes record the comparison; they add no installer gate.

Quality requests use `logprobs=1`, which also rejects the frozen compact alternatives and retains the normal full-vocabulary path. Together with the explicit-seed proof for sixpoint requests, this establishes the serving selection boundary, but does not prove the removed startup work has no numerical or performance effect. Acceptance requires the subsequent precision, state-reuse, restart and performance checks.

Legacy `activate_dense_defaults` only restores `additional_config.mach_dense_auto`. Its original configuration gate requires hidden size 5120, 64 layers, TP2 and FP6 E3M2 weights; this 4B TP1 MXFP8 profile cannot satisfy it. With the disabled legacy producer/prefill/MoE flags, `prepare_model` returns before importing any adapter. The removed calls therefore made no CUDA allocations or launches in the frozen profile. The complete profile must keep those flags disabled and reject `mach_dense_auto=True`. The exact zero-valued legacy flag map is stored as `required_disabled_legacy_environment` in the manifest: `VLLM_MACH_DENSE_AUTO`, `VLLM_MACH_GDN_PERSISTENT`, `VLLM_MACH_GDN_BA_OVERLAP`, `VLLM_MACH_FUSED_SWIGLU_QUANT`, `VLLM_MACH_FUSED_GDN_QUANT`, `VLLM_MACH_FUSED_AR_QUANT`, `VLLM_MACH_MOE_PROJECTION_TUNING`, `VLLM_SM120_OWNER_PREFILL`, `VLLM_SM120_LOSSLESS_PREFILL`, `VLLM_QWEN3_5_FP16_SSM`, and `VLLM_QWEN3_5_FUSED_AR_NORM`. Diagnostic owner/lossless flags have no consumer while their parent modes are disabled. CUDA graph stream warmup now imports Mach's packaged helper; it finds no W6A8 layers and returns before loading a native library or launching work.

The public and frozen `sample/states.py` and `sample/prompt_logprob.py` are byte-identical. Executing the actual seed registration and fast-path predicate on CPU for all six concurrency values confirms the full-vocabulary fallback. With compact hybrid flags false, the frozen penalty state sets both unique-token buffers to `None`; specializing the frozen post-update kernel removes only its unreachable unique-token branch and four unused arguments, producing the public kernel's exact AST. The launch grid and `num_warps=1` are unchanged.

The manifest distinguishes the frozen reference, the pinned upstream sources and the current installed sources. Both 2B and 4B ordered-GDN guards use the current GDN hash; drift detection remains enforced. The separate `fused_recurrent.py` FLA recurrence remains `fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755`. Complete profiles use the resulting hashes recorded in the manifest.

CPU verification covers exact patching, byte/hash guards, dry-run and repeat installation, incompatible packages, locking, symlink rejection, and injected publication failure with rollback. The [October 6 installed-profile qualification](mxfp8-champion.md#validation) adds repeated startup, runtime route/graph receipts, fixed-M gold logprobs and the complete sixpoint comparison.

A sufficient final state check should cover eligible deferred decode, a switch through stock fallback (which must materialize deferred state), new prefill on reused pages, and a return to the original cohort after drain. Include an age wrap across the W4 replay window and M4/M16 transitions. Compare outputs and materialized FP32 state against the accepted path, then repeat the cohort after a fresh worker start. Verify production sixpoint performance separately with production graph policy and compilation choices.
