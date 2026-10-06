# Optional SiLU/block-FP8 warp scheduling

The Mach FP8 profile can select `silu` independently. Set
`VLLM_MACH_FP8_SILU=1` in the worker environment to use the installed native
wheel. With the feature disabled, both patched vLLM call sites return the
original `_C` operator object, preserving the original compiler target.

This is community block-FP8 with FP32 scales, not MXFP8/E8M0. The initial
native boundary is SM120, BF16 contiguous `[M, 18432]` gate/up input,
E4M3 contiguous `[M, 9216]` output, group128 and FP32 `[M, 72]` scales,
`1 <= M <= 2048`, with no scale upper bound. Scales can be contiguous or
transposed with stride `(1, M)`. Other contracts use the original vLLM op.
No additional BF16 rounding occurs between SiLU and multiplication.

`register()` defines the Mach mutation/fake schema without touching CUDA.
`get_fused_op()` selects the original or Mach op, so compiler imports work
before device initialization. The worker calls `verify_runtime()` and
`install()` after its original device setup and before model warmup/capture.
These load a prebuilt `vllm-mach-fp8-activation` wheel for
vLLM0.29.0/Torch2.13.0+cu130/NVCC13.0.88; no request-path compilation occurs. Missing
or incompatible binaries fail startup when explicitly enabled.

The installed profile changes only the group128 activation-fusion target
and the eager wrapper's final op call. The Mach op retains `out`/`scales`
mutation aliases, fake execution and auto-functionalization. It does not
override any `_C` dispatcher registration. `inspect()` reports registration,
native binary SHA256, device and native/fallback/capture invocation counts.
Python capture counts do not count CUDA graph replays.

Prebuild inside the pinned GPU image, from `native/fp8_activation`:

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=2 python setup.py bdist_wheel
uv pip install --no-deps dist/vllm_mach_fp8_activation-*.whl
```

Run `tests/test_fp8_activation_cpu.py` without Torch to verify opt-out identity, metadata selection, registration and worker lifecycle. The SM120 GPU suite passes FP8-byte/FP32-scale comparisons, fallback, changing-input graphs and mutation/fake/actual dynamic-compile contracts. The combined profile's [qualification](fp8-qualification.md) has byte-identical fixed-gold M4/M32/M64 records and successful four-point 3k/1k serving. It does not isolate SiLU's end-to-end gain or establish arbitrary-model/context equivalence.

Historical prototype gains are not combined with the new measurements. The related upstream [#45055](https://github.com/vllm-project/vllm/pull/45055) includes the same warp scheduling direction plus vectorization/ROCm handling. Exact-source upstream component results and acceptance remain separate from Mach model/service qualification.
