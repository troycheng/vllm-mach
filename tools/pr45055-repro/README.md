# PR #45055 exact-head CUDA reproduction

This standalone harness validates the unmodified SiLU/block-quantization source at public vLLM head [`42cff10c75958b8cb1ba1cb991ac8bab9f242aa1`](https://github.com/vllm-project/vllm/commit/42cff10c75958b8cb1ba1cb991ac8bab9f242aa1). The stable-ABI adapter renames namespace and host identifiers and registers `pr45055_exact::run` to avoid symbol interposition with the reference `_C` library. It includes the original kernel and conversion/dispatch helpers from the verified source tree; arithmetic is unchanged. The source snapshot retains its vLLM copyright and Apache-2.0 notices. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The recorded run used SM120, Torch `2.13.0+cu130`, and NVCC `13.0.88`. Its reference was vLLM `0.29.0` from image `vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`, with `_C_stable_libtorch.abi3.so` SHA256 `0375f3ec05f823961476dfe3450cb67df385eb9541eeed29e5daccc28a142ecb`.

[results-summary.json](results-summary.json) records 84 numerical cases and six changing-input CUDA graph replays, all with zero changed quantized bytes or scale bits. The private wrapper also passed auto-functionalization, fake-tensor opcheck, and actual dynamic/fullgraph compilation at rows 3/32/64, with unchanged input bits. The tests compare against the fused reference, preserving its rounding boundary. The copied harness is byte-for-byte identical to the executed source; no seed or numerical algorithm was added. Random samples can differ between reproductions.

Run in the reference image with an SM120 GPU exposed, the matching CUDA toolkit, and `uv` available. Start from this directory. Use the installed image interpreter and expose its reference packages to the venv:

```bash
cd exact_head
uv venv --python "$(command -v python3)" --system-site-packages .venv
.venv/bin/python - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import torch
assert torch.__version__ == "2.13.0+cu130"
assert importlib.metadata.version("vllm") == "0.29.0"
reference = Path(importlib.util.find_spec("vllm._C_stable_libtorch").origin)
assert hashlib.sha256(reference.read_bytes()).hexdigest() == "0375f3ec05f823961476dfe3450cb67df385eb9541eeed29e5daccc28a142ecb"
PY
```

Download and retain the official archive. Verify its identity before unpacking:

```bash
PR_HEAD=42cff10c75958b8cb1ba1cb991ac8bab9f242aa1
curl --fail --location \
  "https://codeload.github.com/vllm-project/vllm/tar.gz/${PR_HEAD}" \
  --output "vllm-${PR_HEAD}.tar.gz"
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
archive = Path("vllm-42cff10c75958b8cb1ba1cb991ac8bab9f242aa1.tar.gz")
assert hashlib.sha256(archive.read_bytes()).hexdigest() == "5debd34ef431cf28c34e85bf50e9cfffa73d59ea92b99895f206689360c01ded"
PY
tar -xzf "vllm-${PR_HEAD}.tar.gz"
export VLLM_PR45055_SOURCE_ARCHIVE="$PWD/vllm-${PR_HEAD}.tar.gz"
export VLLM_PR45055_SOURCE_ROOT="$PWD/vllm-${PR_HEAD}"
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 .venv/bin/python setup.py build_ext --inplace
.venv/bin/python check.py --out results.json
.venv/bin/python -m unittest test_source_identity -v
```

The verifier checks every unpacked file SHA256, directory type and symlink against the archive, rejects missing/extra/changed entries, and separately checks kernel SHA256 `d1741c0a7b4cefee38d0da82505e4bf9914ecc7a31003b0a7b98d710e7936637` against the API snapshot. Archive mode records the public revision and archive identity without inventing a Git HEAD. A pristine checkout at that exact revision is also supported by omitting `VLLM_PR45055_SOURCE_ARCHIVE`. The extension is loaded through `torch.ops.load_library`; it has no PyInit entry point.

On this Torch build, E4M3 `opcheck(test_schema)` fails internally with `"mul_cuda" not implemented for 'Float8_e4m3fn'` for both reference and candidate. The harness records that identical limitation; it does not report schema checking as passed. Its runnable fake and actual compile checks provide separate evidence.

This run did not qualify ROCm, a full PR build or vLLM fusion pass, model/service accuracy, or performance. The exact head's int32 token-offset multiplication remains a static risk: 65536 × 32768 exceeds signed int32 range. No CUDA overflow execution is claimed. Current main's int64 token indexing should be preserved in a future rebase, whose identity must be qualified separately. The candidate binary SHA in the result is a recorded run identity, not a reproducible-build promise.

The public bundle contains source and a compact allowlisted result only. Build outputs may record caller-selected local paths; remove those paths before sharing generated logs. The harness and documentation were prepared with AI assistance.
