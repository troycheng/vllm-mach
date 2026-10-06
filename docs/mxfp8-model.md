# Fixed MXFP8 champion model

`vllm-mach-mxfp8-prepare` reconstructs the fixed
`qwen35-4b-mxfp8-champion-v1` checkpoint on CPU. Reconstruction from the public
BF16 source and the exact L0 asset has passed full file and **595-tensor**
validation in a clean public-base environment. The reconstructed checkpoint also passed [packaged GPU precision and six-point serving qualification](mxfp8-qualification-20261006.md).
See the [complete profile guide](mxfp8-champion.md) for build and serving steps.

## Inputs and access

- **BF16:** `Qwen/Qwen3.5-4B`, immutable revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Keep both safetensors shards,
  their index, config and tokenizer artifacts. The tool verifies full shard
  hashes and the fixed metadata identities.
- **L0 codes:** obtain the exact two-tensor safetensors asset through an
  **authorized channel**. It has not been publicly released; no public download
  URL is available. Required size: `47186176` bytes. Required SHA-256:
  `133aae145c4a21963520f538b2fc3a6cff263f3ccb128ba21ade6d91e63686f4`.
  It contains E4M3 gate/up codes of shape `[9216,2560]`. A PyTorch pickle is
  not an interchangeable input.

The public BF16 revision is identified, but L0 distribution authorization and
access remain unresolved. Do not describe the complete profile as publicly
reproducible until this asset access boundary is closed. Input
hashes verify identity, not redistribution permission; the manifest retains
pending provenance for the L0 asset and complete materialization.

## Prepare and validate

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 vllm-mach-mxfp8-prepare \
  --bf16 /path/to/bf16 \
  --l0-codes /path/to/l0.safetensors \
  --output /path/to/new-checkpoint
```

The output must not exist and its parent must already exist. Inputs must be
ordinary files/directories without symlinks. The tool verifies the fixed
sources, writes to a private temporary sibling, checks every output tensor and
file, then atomically publishes without replacing an existing destination on
Linux/macOS. Failure removes only its own temporary directory.

Reconstruction uses the exact BF16 group32 scale rule and CPU E4M3 casts, with
hash checks for all 200 original codes/scales and 227 retained tensors. It
restores 32 disk q/k/v/o weights in layers `3,7,11,15,19,23,27,31` to BF16,
removes their scales and adds 16 runtime qkv/o quantization ignores. Only the
L0 gate/up codes are replaced; their original scales remain unchanged. The
output contains 595 tensors and retains the tied BF16 embedding/head.

The 81 add-one/BF16/subtract-one norm conversions, 24 linear-attention norm
conversions and retained state dtype conversions are checked against fixed
hashes. Quantization reads 4 MiB source chunks plus arithmetic temporaries;
it does not load the full checkpoint or embedding into memory or require an
intermediate MXFP8 checkpoint.

Standard offline `AutoTokenizer.save_pretrained` converts the BF16 tokenizer,
then exact output hashes are checked. Fixed text-only config, tokenizer config,
chat template and generation config accompany the shard. The demonstrated
conversion uses Transformers 5.16.1; actual Transformers/tokenizers versions
are recorded. No token structures are manually edited or files fetched.

`mach_profile.json` binds the profile to eight full-attention module names and
16 fixed FP32 K/V scales. `mach_materialization_manifest.json` records source
revision/file hashes, output file/tensor hashes, conversion versions and
materializer source hashes, using portable relative file names.

```python
from vllm_mach.mxfp8.prepare_model import prepare_model, validate_model

prepare_model(bf16_dir, l0_codes=l0_file, output=new_output_dir)
metadata = validate_model(new_output_dir, verify_weights=True)
```

Full validation hashes every tensor and shard; the serving launcher performs
it at cold startup. The cheaper default checks headers and metadata binding
and returns `verified_weights=False`. Live worker geometry, storage and
execution still need separate validation.

The older verified-copy path remains available as
`prepare_model(bf16_dir, original_mxfp8_dir, l0_file, new_output_dir)` or
`--mxfp8 /path/to/original-mxfp8`; it is optional and is not needed for direct
BF16 reconstruction. CPU tests run with
`python -m unittest discover -s tests -p 'test_mxfp8_*model.py' -v`.
