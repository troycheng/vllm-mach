# Rebuild the Qwen3.5-2B champion checkpoint

The CPU builder reconstructs the selected 2B MXFP8 checkpoint from the original
BF16 release. It requires no quantized input checkpoint, calibration data, L0
replacement codes, or CUDA device. Install the Mach package and CPU-capable
PyTorch; the builder imports no vLLM GPU runtime.

The source is [Qwen/Qwen3.5-2B at revision
15852e8c16360a2fea060d615a32b45270f8a8fc](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc),
licensed Apache-2.0. Its single 4,548,221,488-byte BF16 shard has SHA256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
This immutable revision's Hugging Face LFS identity matches the source used for
the accepted experiment. Use a real local snapshot directory; source symlinks
are rejected. Keep its config, complete tokenizer.json, tokenizer_config.json,
and chat_template.jinja. The model index must accurately describe the shard.

```sh
vllm-mach-mxfp8-2b-prepare \
  --bf16 /models/Qwen3.5-2B \
  --output /models/Qwen3.5-2B-MXFP8-BF16AttentionEnds
```

The output parent must exist. The output directory must not exist and must be
separate from the source. The builder checks the pinned source identities,
streams bounded CPU tensor rows, checks all reconstructed tensor payloads, and
publishes the directory only after full validation. Failure removes only the
temporary directory it created. A concurrent existing output is never replaced.
Allow space for the 3,630,932,360-byte output shard alongside the BF16 source.

The output contains 438 tensors. Group32 E4M3 values and uint8 E8M0 scales use the
audited BF16 observer and rounding recipe. RMS weights retain the original
FP32 +1, BF16 rounding, FP32 -1 conversion; linear-attention norm weights retain
their BF16 conversion. The tied BF16 head is reconstructed from the original
embedding weight. All full-attention q/k/v/o projections, layer0 GDN qkv/z/out
and MLP gate/up/down, and layer23 MLP gate/up/down stay BF16: 33 weights in
total. Their scales are absent and the quantization ignore list includes fused
module aliases. The complete BF16 tokenizer and chat template are copied,
without modifying vocabulary or truncating tokenizer.json.

The bundled `two_b/data/qwen35-2b-model.json` binds the source identities, exact
output config, original safetensors header, and every tensor payload SHA256.
Preserving the original header and offsets also reproduces the entire accepted
checkpoint file, rather than only equivalent tensor contents. Its SHA256 is
`115948c161e2cc8d02a4e7e580cefcf3c02513410a2af56e80dd63f9f48a1671`.

Cold startup can validate an independently copied accepted checkpoint or a newly
rebuilt one with the same API:

```python
from vllm_mach.mxfp8.two_b.model import validate_model

identity = validate_model("/models/Qwen3.5-2B-MXFP8-BF16AttentionEnds",
                          verify_weights=True)
assert identity["tensor_count"] == 438 and identity["verified_weights"]
```

`verify_weights=False` checks tensor headers, config, and tokenizer identities;
it does not certify weight payload integrity. Full validation hashes all 438
tensors and the complete shard. The build also records `mach_2b_model.json`.
This checkpoint builder does not install or select the serving optimizations;
the 2B launcher must separately activate native MXFP8, ordered GDN, pinned reset
IDs, BF16 BA, and PW2048 with the accepted 2B runtime contract.

Run the CPU tests with:

```sh
python -m unittest discover -s tests -p test_mxfp8_2b_model.py -v
```

The arithmetic test requires PyTorch; the other tests cover source drift,
metadata and tokenizer drift, payload verification, source alias ambiguity,
and failure/publication handling without a GPU. Full-model CPU reconstruction
is the release check for the pinned 438-tensor identity.
