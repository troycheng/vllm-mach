# FA2 mixed-batch exact-source diagnostic

[vLLM PR #58013](https://github.com/vllm-project/vllm/pull/58013) belongs to LiRunGuo. Mach tested its unchanged source at `e17555d2673e632cbe85365a53490a138e4b2fae` against the recorded base `382970ee6ca490aeaaaf4e32c53695b581ff61ba`, with compatibility bindings to vLLM 0.29.0 / Torch 2.13.0+cu130 and its FA2 binary on RTX 5090 / SM120.

The test directly executes those sources' metadata, build and forward bodies, their dense-call wrapper and exact-head split helper. Model-independent builder state is initialized by the driver; upstream constructors, the complete upstream runtime and the model runner are outside this check. There is no model-quality or serving-performance result. See the [compact observed receipt](data/fp8-fa2-source-check-20261006.json).

## Public synthetic result

[The public driver](../tools/verify_fa2_mixed_head.py) was run on GPU with seed `20261006`. It generates BF16 inputs with 16 query heads, 4 KV heads, D256 and 528-token KV pages, using a 64-block shared random cache bank and KV length 3072. Both inputs have 2048 query tokens and 8,388,608 BF16 output values. The c32 case has 30 one-token decodes plus prefills of 526 and 1492 tokens; c64 has 60 one-token decodes plus prefills of 646 and 1342 tokens. No checkpoint, prompt corpus or private fixture is required.

The base makes one call with `num_splits=0`; the unchanged head makes two subcalls with requested/effective `num_splits=0`:

| Nominal concurrency | Different BF16 values | Prefill differences | Whole-output MAE | Maximum absolute difference |
|---|---:|---:|---:|---:|
| c32 | 52,260, all decode | 0 | 5.93755e-7 | 0.0009765625 |
| c64 | 100,507, all decode | 0 | 1.12464e-6 | 0.0009765625 |

A separately labeled diagnostic keeps the head source unchanged and overrides only its FA2 subcalls to effective `num_splits=1`; the base retains zero. Both inputs then have zero differing values. Pure decode and pure prefill controls are exact, output padding is preserved, and the finite-output and call-count assertions pass.

The changed numerical result follows the diagnostic split-count strategy on these inputs; the underlying kernel/accumulation cause remains unconfirmed. This is not evidence of a full-model accuracy loss, and the diagnostic override is not a passing result for the original head. The driver does not force mixed CUDA graph capture or measure performance. A COMPLETE receipt can contain original-head numerical differences; it does not imply byte equality.

## Reproduce

In a compatible Torch 2.13.0+cu130 / vLLM 0.29.0 GPU environment that selects FA2:

```bash
python tools/verify_fa2_mixed_head.py --seed 20261006 --output ./fa2-synthetic-result.json
```

The driver fetches pinned official base/head/helper Python sources and verifies their SHA-256 hashes before extracting the unchanged methods. `--sources` can point at a prepopulated cache with `base_flash_attn.py`, `head_flash_attn.py` and `head_utils.py` for an offline run. Their required hashes are in the script. `--seed` changes only synthetic inputs; exact counts can depend on the compatible runtime and GPU. `--fixtures` is optional for users who have their own compatible captured data.

The receipt records original-head and diagnostic-override results separately, including requested/effective split counts, changed values, MAE and maximum absolute error. AI assistance was used to prepare the driver and documentation; the reported numbers come from its actual GPU execution.

## Captured-input supplement

Two separate captured attention inputs show the same boundary: the recorded base reproduces their saved outputs exactly; the unchanged head differs in 10,538 / 19,506 decode values at c32/c64, with maximum absolute differences 0.0078125 / 0.015625. Prefill differences are zero. The diagnostic override to one again restores byte equality, and pure-input/padding controls pass. Those operands are not distributed and are not required for the public result above.

The immutable captured receipt embeds an old pre-run "Not executed" provenance string. The compact report distinguishes that text from the completed execution verdict and keeps the captured and synthetic results separate.
