# Quantization sources

Generation helpers for the Qwen3.8-27B Champion profile. Start with `python3 tools/generate_assets.py --help` and the [recipe and dependencies](../../docs/quantization.md). The existing numerical primitives are unchanged; the orchestration has CPU/interface checks, not a repeated full-model generation run.

| File | Existing entry point | Input → output |
| --- | --- | --- |
| `offline_quant.py` | `optimize_and_pack(weight, hessian, weight_container)` | Original BF16 merged gate/up and QA block Hessian → packed NVFP4 weight |
| `weight_container.py` | `DirectNVFP4Weight` | Packed values, scales and geometry → storage object used by the packer |
| `fit_rank64.py` | `fit(error, train, tensor_sha256)` | BF16-to-NVFP4 weight residual and mixed QA/code inputs → BF16 low-rank factors |
| `derive_static_scales.py` | `--model-dir PATH --out FILE` | Original RMSNorm weights → static activation-scale receipt |
| `calibration-selection.json` | Metadata, not executable | Exact sample IDs, windows, splits and source hashes |
| `upstream/` | Source references checked by `offline_quant.py` | Frozen NVIDIA Model Optimizer scale-selection sources |
| `prepare_calibration.py` | `prepare` | Pinned LongBench archive and tokenizer → fixed token manifests |
| `capture.py` | CLI used by the generator | Current Mach runtime and token manifest → BF16 M32 inputs |
| `build_rank64.py` | CLI used by the generator | BF16 model and captures → importable rank64 bundle |

The Python files retain the original arithmetic. `weight_container.py` contains only the original storage class, extracted from a larger conversion module; it does not include that module's EXL3 reconstruction path. `sources.json` records where each file came from. Nothing here is imported by the serving runtime.

`teacher_decode.py` and `chunked_teacher.py` adapt the original diagnostic-only teacher forcing to the pinned vLLM 0.29 sampler identities. They are installed only in the generation subprocess, never by the serving launcher. `capture.py` exposes the BF16 output of the fused AR/RMSNorm operation and observes the current Mach gate/up weight call. QA and code captures finish before packing begins.

The static-scale extractor can also be called independently:

```bash
python3 tools/quantization/derive_static_scales.py \
  --model-dir /path/to/original-bf16-model \
  --out /path/to/new-static-scales.json
```

It uses Python 3.11+ standard-library APIs and reads the selected RMSNorm tensors, not whole weight shards. Source paths and script hashes may change in a new receipt. The importer checks the numerical content and norm identities against the selected model, and records the new receipt's file hash in the bundle.
