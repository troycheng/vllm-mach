# Install the MXFP8 runtime profile

The 2B and 4B MXFP8 profiles use Mach 0.3, vLLM 0.33.0 and the shared [runtime versions](installation.md#runtime-versions). `deploy/Dockerfile.mxfp8` builds the native MXFP6/MXFP8 library, installs the head and BA dependencies, then applies the runtime profile.

For an existing matching environment:

```sh
python -m pip install --no-deps --no-build-isolation .
python -m vllm_mach.mxfp8.install
python -m vllm_mach.mxfp8.install --apply
```

The first installer command previews changes; `--apply` installs them. Stop the serving worker before applying an update, then restart it. `--site /path/to/site-packages` selects another installation directory. The Python API is `install_profile(site, apply=False)`; `inspect_sources(site)` checks installed sources without initializing CUDA.

## Runtime changes

Mach packages a shared overlay in `runtime/common-vllm033.patch` and the MXFP8-specific overlay in `mxfp8/data/runtime.patch`. The manifest records their checksums, the upstream commit, and the original and installed file hashes. The shared overlay is also used by native MXFP6; each profile adds its own adapter and model configuration.

| Area | Integration |
|---|---|
| GDN / recurrent state | SM120 prefill and FLA routing, ordered state updates, selected state initialization, state-index and mixed-batch metadata |
| FlashInfer | SM120 XQA eligibility, FP8 KV layout and calibrated scale handling |
| Worker lifecycle | Workspace preparation, graph capture routing and model-specific initialization on the vLLM 0.33 runner |
| Sampling and head | Model-specific full-head and hybrid-head hooks, current sampler state and fallback handling |
| Compilation | Explicit model/mode bindings, launch configurations and reduction order |

Both profiles retain their published model, SSM, head, KV and scheduling settings. The 2B launcher selects its own attention and graph policies; the 4B launcher installs its calibrated KV scales and complete K2048 head. Use the [2B guide](mxfp8-2b-champion.md) or [4B guide](mxfp8-champion.md) for model preparation and serving.

## Installation checks

The installer validates package versions and input source hashes, applies the common and profile patches in staging, and checks the resulting hashes and Python syntax. It then publishes complete files under an installer lock. A failed publication restores the replaced files; reapplying the same profile leaves them unchanged.

At startup, worker receipts record installed source identities, native libraries, compiler bindings, graph policy, model geometry and precision settings. Native wheels are compiled during image construction; model-specific Triton kernels and graphs are prepared during worker startup.
