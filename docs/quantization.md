# Generating the model assets

Mach's current Qwen3.8-27B profile uses an EXL3 checkpoint, an original-model MXFP6 checkpoint, and an optional NVFP4/rank64 bundle. Mach does not upload these model assets. `tools/generate_assets.py` provides two generation commands using the existing quantizers. The assembled workflow has CPU/interface checks, but has not been rerun through full-model conversion and calibration.

The `--model` entry point remains the K5/K6 EXL3 checkpoint. The additional `--mxfp6-checkpoint` supplies original-model MXFP6 weights for the [checkpoint-hybrid routes](checkpoint-hybrid.md); it is not a cache converted from EXL3. The selected M32 gate/up route uses the separate `--rank64-bundle`. Calling this an EXL3 profile identifies the provider and entry checkpoint, not the format of every matrix it executes.

## MXFP6 checkpoint

The existing [llm-compressor fork](https://github.com/troycheng/llm-compressor/tree/37242a6a1bf6869857084d2ac7ccb22d1af7168d) provides the model-free MXFP6 converter. Its public [Qwen Dense example](https://github.com/troycheng/llm-compressor/blob/37242a6a1bf6869857084d2ac7ccb22d1af7168d/examples/quantization_w6a8_mxfp6/qwen35_27b_example.py) shows the same `model_free_ptq` API, but targets Qwen3.5. The Qwen3.8 settings below are taken from the existing checkpoint configuration, not from a newly validated conversion.

Install the converter in a separate environment, keeping its dependencies out of the serving image:

```bash
python3 -m venv quant-env
. quant-env/bin/activate
python3 -m pip install \
  'git+https://github.com/troycheng/llm-compressor.git@37242a6a1bf6869857084d2ac7ccb22d1af7168d'
```

Download the original BF16 model from `Qwen/Qwen3.8-27B` on ModelScope at revision `e823e888ae179eb3be02c1a48899c4f828371376`. The generator checks its config/index identities before starting. Keep input and output directories separate. From the Mach checkout, run:

```bash
python3 tools/generate_assets.py mxfp6 \
  --model /path/to/Qwen3.8-27B \
  --output /path/to/Qwen3.8-27B-MXFP6 \
  --device cuda:0
```

The wrapper calls `model_free_ptq(scheme="MXFP6", max_workers=1)` with the exclusions recorded in the existing checkpoint. Add `--dry-run` to inspect the resolved arguments without importing the converter or quantizing weights. The output directory must be new. Keep enough disk space for both the BF16 input and approximately 26 GB of output.

Weights use E3M2 with one UE8M0 scale per group of 32, serialized in Quark/OCP-MX format. Activations are quantized dynamically to MXFP8 at serving time. This conversion requires no calibration dataset. It quantizes the original model, not reconstructed EXL3 weights. The excluded embeddings, output head, small GDN A/B projections, visual tower and MTP modules remain dense.

The converter also supports CPU execution. Its native and portable paths, dependency versions and output serialization may differ from the historical run; the settings above are a generation recipe, not a claim of newly confirmed byte-identical output. The original conversion environment was not fully recorded in the exported checkpoint.

## NVFP4 weights and rank64 compensation

Run this stage in the [Mach image](installation.md#build-the-image), which supplies the patched vLLM 0.29.0 runtime and native dependencies. Download `data.zip` from the pinned LongBench revision linked below. The script checks the archive, tokenizer and each selected token window; it does not download or choose a different dataset.

Assuming the original BF16 model, EXL3 model and MXFP6 output are in `models/bf16`, `models/exl3` and `models/mxfp6`, and the archive is in `calibration/data.zip`:

```bash
mkdir -p generated
docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/calibration:/calibration:ro" \
  -v "$PWD/generated:/generated" \
  -v mach-kernel-cache:/root/.cache \
  --entrypoint python3 vllm-mach:local \
  /opt/mach/tools/generate_assets.py rank64 \
  --model /models/bf16 --exl3 /models/exl3 --mxfp6 /models/mxfp6 \
  --calibration-archive /calibration/data.zip \
  --work-dir /generated/work --output /generated/rank64
```

This command schedules two sequential TP2 capture subprocesses (QA, then code), followed by layer-by-layer weight packing and fitting on CUDA device 0. It does not run throughput benchmarks. Choose two idle SM120 GPUs with working P2P. Keep the work directory to inspect the capture manifests and fit records; intermediate weights and final bundle together require roughly 11 GB plus activation captures and kernel caches. Work and output directories must be new. `--dry-run` checks the BF16 model config/index and prints the subprocess commands without preparing data or using a GPU.

Use `generated/rank64` as `--rank64-bundle` in the serving command. The EXL3 and MXFP6 model directories remain separate; this change does not introduce a new checkpoint format.

The [collected scripts](../tools/quantization/README.md) preserve the selected algorithm. The order matters:

1. Capture BF16 inputs to gate/up in the checkpoint-hybrid runtime, using the fixed QA samples. Form a block-diagonal Hessian from training rows only.
2. Quantize original BF16 gate/up weights to NVFP4 using that QA Hessian. Retain the selected 48 layers and both TP shards.
3. Capture the additional long-code training inputs. Fit rank64 compensation against the already selected NVFP4 weights using concatenated QA/code rows. Do not requantize the weights with the mixed dataset.
4. Derive static activation scales from the original RMSNorm weights.
5. Package the resulting tensors and receipts with [import_rank64_bundle.py](../tools/import_rank64_bundle.py).

### Fixed geometry and calibration

The selected layers are `0–26, 34, 35, 36, 38, 42, 48–63`. For each TP rank, take 8,704 output rows from each original gate and up matrix, concatenate gate then up, and obtain BF16 `[17408, 5120]`. This produces 96 weight entries across 48 layers and TP2.

[calibration-selection.json](../tools/quantization/calibration-selection.json) records the exact source IDs, token windows, train/holdout membership and hashes. Source data is [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672), revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. The original tokenizer JSON hash is `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; the long-code preparation used tokenizers 0.22.2. Source texts and captured activations are not included here.

| Input set | Capture | Training rows per layer/rank | Use |
| --- | --- | ---: | --- |
| QA | 32 samples, 192 prompt + 8 teacher-forced tokens; offsets 1–8 | 16 train samples × 8 offsets = 128 | NVFP4 Hessian and rank64 fit |
| Long code | 32 samples, 3000 prompt + 1000 teacher-forced tokens; offsets 1,128,256,384,512,640,768,1000 | 16 train samples × 8 offsets = 128 | Rank64 fit only |

The QA set covers eight tasks, with two training and two holdout samples per task. Long code uses `lcc` and `repobench-p`, with eight training and eight holdout samples per task. Calibration uses eager physical-M32 decode and BF16 gate/up inputs, with fused AR/RMSNorm/MXFP8 disabled to expose that input. The original capture used FP16 recurrent state. Long-code capture used 4096-token prefill batches and a per-request chunk threshold of 128. Holdout rows are not used to fit the weights or compensation.

### Weight packing

`offline_quant.optimize_and_pack` preserves the Hessian-selected scale sweep adapted from [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/51de53e48ccae8804f8fe1198b7cf89475c5c4f4). For each 16-element input block, `H = XᵀX / 128`. It selects an E4M3 block scale by minimizing `ΔwᵀHΔw` over the fixed positive finite E4M3 candidates. The global reciprocal scale is `2688 / max(abs(W))`.

The output stores low-nibble/even-K and high-nibble/odd-K E2M1 codes, with FlashInfer's 128×4-interleaved block scales. Replacing this step with an ordinary absmax NVFP4 quantizer discards the selected scales. The existing implementation uses PyTorch, Triton and FlashInfer; the runtime port was tested with PyTorch 2.13.0 and FlashInfer 0.6.18, while the original calibration environment was separate.

### Compensation and static scales

`fit_rank64.fit` takes `Δ = W_BF16 − dequantize(W_NVFP4)` and the 256 concatenated BF16 training rows. It computes `Y = XΔᵀ`, obtains the top 64 output-space singular directions through the FP64 Gram eigendecomposition, and stores `A = ΔᵀV` and `B = Vᵀ` in BF16. The existing function returns ranks 32 and 64; this profile uses only rank64. Its `tensor_sha256` argument is a caller-supplied checksum function, not a calibration parameter. TF32 was disabled during fitting.

The runtime combines dual-A4 NVFP4 execution with `XAB`. Factor shapes are `[5120,64]` and `[64,17408]`. This is a quantization-error correction, not a LoRA adapter or a lossless representation of the original BF16 model.

`derive_static_scales.py` reads RMSNorm weights and selects a power-of-two activation scale from `2688 / (1.01 × sqrt(5120) × max(abs(1 + norm_weight)))`. The selected profile uses 16 or 32. This step does not use calibration data.

## Integrity and validation

Capture receipts bind each TP payload to its calibration manifest. The builder uses training rows only and binds compensation factors to the NVFP4 tensors they correct. Packing retains the existing 48-layer mask and numerical kernels.

Static-scale validation checks all 64 RMSNorm weight hashes, selected scales and model geometry against the original numerical contract. New bundles record their own report-file hash; source paths and script-location metadata may change. Legacy bundles retain their original file-hash check. The model config/index, tensor shape/dtype and weight-to-compensation checks remain in place.

The orchestration and vLLM 0.29 capture adapter have not been validated by regenerating the full model. Existing Champion correctness/performance results apply to the existing assets, not automatically to newly generated assets. No generation job or new performance test was run while adding these commands.
