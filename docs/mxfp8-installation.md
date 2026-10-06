# Install the MXFP8 runtime source profile

The `qwen35-4b-mxfp8-champion-v1` profile installs a reviewed source delta on the public vLLM 0.29.0 runtime. It does not require the original experimental image. Install the Mach wheel and its native projection/head dependencies separately, then run this installer before starting workers:

```sh
python -m vllm_mach.mxfp8.install
python -m vllm_mach.mxfp8.install --apply
```

The first command is a dry-run. `--site /path/to/site-packages` selects a directory containing `vllm/`; package-version checks still apply. The direct Python API is `install_profile(site, apply=False)`. Offline source audits can pass `verify_packages=False`; the CLI has no such bypass. `inspect_sources(site)` verifies an installed source tree without importing vLLM, Torch, Triton, or CUDA.

`data/runtime_sources.json` records upstream, resulting, and frozen comparison SHA256 values, the bundled patch SHA256, package versions, file purposes, and the formalization boundary. It contains no model references, old workspace paths, or copied experiment receipts. Required runtime packages are vLLM 0.29.0, Torch 2.13.0, Triton 3.7.1, FlashInfer Python/cubin 0.6.18, CUTLASS DSL 4.6.2, and CUDA bindings 13.3.1. Native MXFP8/dual/BA and NVFP4 head dependencies remain the complete profile's responsibility.

The installer accepts exactly the pinned public source tree or its own resulting profile. It rejects changed source bytes, a partial installation, symlinks, incompatible versions, and a corrupt bundled patch. Every hunk applies at its stated offset with exact context; offsets and fuzz are never accepted. All resulting hashes and Python syntax are checked in staging before any source is replaced. Applying an already installed profile makes no changes.

Apply to a stopped runtime. Publication takes an exclusive installer lock, rechecks the staged input bytes, replaces each whole file atomically on its filesystem, and restores already replaced files if a later publication fails. Permissions are preserved. This provides rollback for reported installation failures, not a filesystem transaction across process crashes or power loss. The installer builds no MoE, allreduce, IPC, CUDA extension, or model library.

## Source boundary

The public base plus Mach's existing MXFP6 common patch was applied locally with zero fuzz, then compared with the frozen runtime. It was insufficient: the GDN and FlashInfer changes are larger than import substitutions. The independent MXFP8 patch changes seven files and pins sixteen runtime sources:

| Source | Result |
| --- | --- |
| `envs.py` | Frozen defaults, flags, and compilation-factor definitions |
| `utils/flashinfer.py` | Frozen XQA eligibility, autotuning, and quantization helpers |
| `qwen_gdn_linear_attn.py` | Complete frozen source, byte-for-byte; includes the SM120 prefill/FLA gates and custom-op `hidden_states` boundary |
| `v1/attention/backends/flashinfer.py` | Complete frozen source, byte-for-byte; separate K/V head storage, FP8 KV update, XQA metadata and decode |
| `v1/worker/gpu/cudagraph_utils.py` | Frozen capture candidates/order and capture stream; package import for the empty MXFP6 warmup |
| `v1/worker/gpu/model_runner.py` | Public full-vocabulary runner with the frozen prompt-logprob profiling warmup |
| `v1/worker/gpu_worker.py` | Frozen graph-memory/lifecycle behavior; removes verified disabled legacy MXFP6 adapter calls |
| Public logits processor, Qwen3.5 model, sampler/states/penalties/input batch, prompt-logprobs, FLA recurrence and MXFP8 FlashInfer linear kernel | Exact public source guards |

The installer omits the legacy model runner, MoE modifications, and both native allreduce source changes. This profile requires the v2 runner, a dense MXFP8 4B model and TP1, with AR fusion disabled. It does not install the MXFP6 profile's MoE/AR/IPC additions.

## Arithmetic and operative changes

GDN and FlashInfer computational source bytes are unchanged from the frozen runtime. CUDA graph candidate selection and capture streams remain frozen. The prompt-logprob profiling helper and its launch are retained, including the ordinary prompt-logprobs serving entrypoint used for quality checks.

The public logits processor's `forward`, `_apply_head`, `_get_logits` and `_gather_logits`, and the public sampler's `apply_sampling_params` and `sample`, have identical ASTs to their frozen counterparts. The formal NVFP4 full-vocabulary head replaces `_apply_head` after compile; it does not need the frozen compact logits helpers. Compact hybrid modes and vocab-parallel greedy are disabled. The fixed sixpoint requests carry explicit seeds; the frozen sampler's explicit-seed guard rejects its compact alternatives, so it reaches the normal full-logits sampler. The public runner's normal complete-vocabulary branch is retained, rather than importing unavailable compact sampling helpers. This is an operative source simplification, not an assertion that arbitrary compact/unseeded workloads are supported.

The public `vllm.v1.worker.gpu.warmup` source has SHA256 `ca8e7bdcc0edd429904c8c509f7bdb5327e02824eebd24cb8886460568d75682`; the frozen source has SHA256 `def9c07eb673b2fd479edf4ddf48483c267563f6eb8dfa5cb6a6cc608bdb6f1e`. The frozen version appends three real request/sample/cleanup warmups for greedy, full-distribution and top-k sampling, gated by model helper methods. The public version omits that block and its Qwen model omits those helpers. The accepted startup log confirms all three compact fast paths ran despite the disabled hybrid/greedy flags: these warmup requests are unseeded and request no logprobs. Removing them is an operative startup change, not a no-op or a claim that all warmup launches are equivalent. These hashes record the comparison; they add no installer gate.

Quality requests use `logprobs=1`, which also rejects the frozen compact alternatives and retains the normal full-vocabulary path. Together with the explicit-seed proof for sixpoint requests, this establishes the serving selection boundary, but does not prove the removed startup work has no numerical or performance effect. Acceptance requires the subsequent precision, state-reuse, restart and performance checks.

Legacy `activate_dense_defaults` only restores `additional_config.mach_dense_auto`. Its original configuration gate requires hidden size 5120, 64 layers, TP2 and FP6 E3M2 weights; this 4B TP1 MXFP8 profile cannot satisfy it. With the disabled legacy producer/prefill/MoE flags, `prepare_model` returns before importing any adapter. The removed calls therefore made no CUDA allocations or launches in the frozen profile. The complete profile must keep those flags disabled and reject `mach_dense_auto=True`. The exact zero-valued legacy flag map is stored as `required_disabled_legacy_environment` in the manifest: `VLLM_MACH_DENSE_AUTO`, `VLLM_MACH_GDN_PERSISTENT`, `VLLM_MACH_GDN_BA_OVERLAP`, `VLLM_MACH_FUSED_SWIGLU_QUANT`, `VLLM_MACH_FUSED_GDN_QUANT`, `VLLM_MACH_FUSED_AR_QUANT`, `VLLM_MACH_MOE_PROJECTION_TUNING`, `VLLM_SM120_OWNER_PREFILL`, `VLLM_SM120_LOSSLESS_PREFILL`, `VLLM_QWEN3_5_FP16_SSM`, and `VLLM_QWEN3_5_FUSED_AR_NORM`. Diagnostic owner/lossless flags have no consumer while their parent modes are disabled. CUDA graph stream warmup now imports Mach's packaged helper; it finds no W6A8 layers and returns before loading a native library or launching work.

The public and frozen `sample/states.py` and `sample/prompt_logprob.py` are byte-identical. Executing the actual seed registration and fast-path predicate on CPU for all six concurrency values confirms the full-vocabulary fallback. With compact hybrid flags false, the frozen penalty state sets both unique-token buffers to `None`; specializing the frozen post-update kernel removes only its unreachable unique-token branch and four unused arguments, producing the public kernel's exact AST. The launch grid and `num_warps=1` are unchanged.

The manifest distinguishes unchanged frozen arithmetic, operative cleanup, and the package-import change. GDN's resulting SHA remains `8ea18f04ee77359e8dea3e089bb991219468f7985648dc4e24eacb9cd20cf719`; the FLA recurrence remains `fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755`. The formal v2 runner has a new SHA recorded in the manifest, which the complete profile's source contract must use.

CPU verification covers exact patching, byte/hash guards, dry-run and repeat installation, incompatible packages, locking, symlink rejection, and injected publication failure with rollback. Release validation must still pair clean starts of the frozen and public-base rebuilt profiles: compare prefill/decode and slot-reuse outputs, graph shape/mode/stream receipts, prompt-logprobs, and the fixed sixpoint contract. CPU checks establish installability and source boundaries; they do not certify GPU output equivalence.

A sufficient final state check should cover eligible deferred decode, a switch through stock fallback (which must materialize deferred state), new prefill on reused pages, and a return to the original cohort after drain. Include an age wrap across the W4 replay window and M4/M16 transitions. Compare outputs and materialized FP32 state against the accepted path, then repeat the cohort after a fresh worker start. Verify production sixpoint performance separately with production graph policy and compilation choices. Repeating one cohort exactly only establishes determinism within that run; it does not establish equivalence to the accepted runtime.
