# Native MXFP6 integration

This profile ports the optimization series based on vLLM `v0.29.0`
(`98dff2a81d747d1dba01a47f939f48c3526d4206`), ending at
`8983c0369eb5dfae42194261d06d8fbbdaab15e6`.
The optimization checkout is not a runtime dependency.

## Imported boundaries

| Source commits | Mach integration |
|---|---|
| `b4ac4e3d7a`, `e77a2e5dc3` | Native dense kernel, Quark MXFP8 metadata, accurate backend reporting, both runners' graph/workspace lifecycle |
| `6b57b3ba2d` | Qwen TP2 AllReduce/residual/GemmaRMSNorm, including final norm |
| Mach `8c5021a` | Persistent small-batch GDN and BA overlap port, packaged with null-slot-zero compatibility |
| `9fb75fd71c` | Narrow FP16 SSM admission for native packed GDN decode |
| `9f1c8fec69`, `7c688885a7`, `b032cc235b` | Lossless prefill dispatch, GDN graph break, warmed opt-in codec graphs |
| `b7b4fb089e`, `f7688f8907`, `a2ce755fde` | Owner row transport, replicated MLP, shape-dependent dispatch down to 512 rows |
| `ff180d2371`, `412d11fb26` | NVFP4 LM head candidate search/refinement, compact greedy sampler and request fallbacks |

Mach reuses its existing native dense kernel and warmup implementation. The
general plugin registers the kernel; the source patch does not import Mach from
vLLM's linear registry, avoiding an import cycle. The framework modules
live under `vllm_mach.mxfp6`. GEMM CUDA implementations remain in `mxfp6-sm120`, with prefill code in
the `native/lossless_prefill` / `native/owner_prefill` wheels, now required
for Dense default (unless both prefill options are explicitly disabled).
The persistent GDN source is packaged under `vllm_mach.mxfp6.gdn` and JIT-built
through FlashInfer during warmup.

The wheel includes a 13-file dense source patch, a separate Quark MoE patch,
a version/file manifest and
FlashInfer local IPC patch. `vllm-mach-install` checks the official dependency
versions, stages both patches and checks patch applicability before writing.
Incompatible or partially patched installations are rejected. Repeat installation
is a no-op. The source patch and plugin must both
be installed; the launcher checks the profile before starting workers.

The EXL3 implementation, old runtime overlays and offline asset tools are removed
and remain available in Git history. The default pytest suite targets
the current native MXFP6 package, warmup and deployment paths.

## Numerical and execution contracts

The base path retains packed checkpoint weights and dynamically quantizes
activations to MXFP8. Manual AR/Norm fusion changes reduction/rounding order.
The launcher uses BF16 activations/KV, TP2/PP1, V2 runner, non-speculative
text-only dense inference and full decode graphs at 1/2/4/8/16/24/32 rows.

The default launcher also enables persistent GDN at physical M1/2/4/8 and
BA overlap at M16/24/32. Persistent changes arithmetic and supports FP32 or FP16 SSM storage
with FP32 accumulation. Both routes have independent opt-outs and
retain native execution for unsupported calls. See [GDN validation](gdn-decode.md)
and the [current measurements](native-fidelity.md).

FP16 SSM is opt-in because it changes state precision. Owner prefill retains
the source series' geometry checks and numerical verification mode; it chooses
communication and MLP paths by row count. Lossless codec graph capture is
implemented but disabled in the default serving profile.

The optional NVFP4 head retains the checkpoint BF16 head, creates an NVFP4 copy,
selects 128 candidates, refines with BF16 weights/FP32 accumulation and exchanges
TP argmax pairs. This approximate search can miss the BF16 winner and rounding
can change near ties. It accelerates eligible greedy decode only. Full logits,
logprobs, stochastic sampling, processors, structured outputs and mixed/prefill
batches retain the normal path. The default compact BF16 argmax uses the original
head projection without NVFP4 search.

## Validation

Use a clean runtime after [installation](installation.md):

```bash
CUDA_VISIBLE_DEVICES=0,1 MXFP6_AUTOTUNE=off python -m pytest -q
```

The tests cover staged installation, idempotency, rejection without writes,
EXL3-free plugin registration, checkpoint scale packing, changing-input graph
replay, request eligibility, full-logit preservation, FP16 admission guards,
prefill shape/workspace contracts, and optional native TP2 transport/codec
comparisons. GPU and optional-extension tests skip when prerequisites are absent.

The September 16 GDN port passed all **168 current tests**, checked in groups,
including CPU routing/data checks and GPU graph/prefill checks. The standalone
persistent harness additionally passed 16 FP32/FP16 × SD/DS × M1/2/4/8 cases with
changing inputs, slot reuse, null-slot protection and bitwise graph/eager
agreement. The rebuilt wheel includes the GDN CUDA source and matches the
package source. [Measured serving/fidelity results](native-fidelity.md) and
[individual GDN ablations](gdn-decode.md) are recorded separately.

The final September 15 rerun passed **113/113 tests**, with no skips, including
the published four-profile serving counts/throughput, fidelity bootstrap and
head-recall checks. It ran on SM120 GPUs 4/5 with the optional extensions
installed. Dependency deprecation warnings were reported; there were no test
failures. The final wheel contains only the native provider, matches the source
package files, and passes the installed-profile preflight without changes.

Performance measurements from the original optimization checkout are not
automatically claims about this packaged integration. Use equal model, topology,
request shape, graph configuration and KV allocation when comparing runs.

### Integration acceptance (2026-09-15)

The initial native test suite passed **102/102**, including real SM120 graph replay,
TP2 lossless codec comparisons, owner norm/transport comparisons and NVFP4
full-logit fallbacks. The installer was also applied to newly downloaded
official vLLM 0.29.0 and FlashInfer 0.6.18 wheels: the first application passed,
and the second changed no files. The serving environment's Python, CUDA source,
headers and shared libraries were compared against those patched official
wheels with no differences. No optimization checkout was on its import path.

The acceptance service completed 64/64 requests at 3000 input / 128 output
tokens with all optional switches and prefill verification enabled. Four
non-thinking Chinese chat checks returned the expected arithmetic, capital,
translation and chemical-formula answers. Nine completion probes exercised
greedy, logprob, stochastic and token-bias requests successfully. These are
smoke checks, not a broad model-quality evaluation or a bitwise-equivalence claim.

This earlier integration acceptance uses uniform token-ID prompts; the README's
current [profile comparison](native-fidelity.md) uses frozen ShareGPT
prefixes and separately validated stock baselines. Do not combine the two
workloads' throughput numbers.

Acceptance serving comparison: two RTX 5090 GPUs, TP2/PP1, BF16 activations/KV, V2 runner,
TRITON_ATTN, no prefix cache, 4096 scheduled tokens, maximum length 16384,
graph sizes 1/2/4/8/16/24/32, fixed KV allocation 8218214400 bytes/rank.
Each arm used a 32-request warmup followed by 64 scored requests at concurrency
32, with 3000 input / 1000 output tokens, greedy sampling and ignored EOS.
Both retained arms completed 192000 input and 64000 output tokens, with no
preemption warnings. Runs were sequential on the same GPU pair, once per arm.

| Configuration | Output token/s | Mean TPOT | BS32 decode iteration median |
|---|---:|---:|---:|
| Default Mach profile: AR/Norm + compact BF16 greedy TP argmax | 1419.6 | 19.81 ms | 13.97 ms |
| Full Mach profile: AR/Norm, FP16 SSM, lossless/owner prefill, NVFP4 head | 1646.8 | 17.20 ms | 12.25 ms |

The historical default profile above retained FP32 recurrent state without
prefill extensions or NVFP4 search. The current Dense default enables both
prefill extensions. The full profile uses FP16 and approximate
NVFP4 head search. This measures the combined profile, not an isolated kernel
gain. Iteration medians include warmup; request throughput and TPOT exclude it.
One run per arm does not establish a confidence interval.

[Machine-readable configuration and results](data/native-mxfp6-acceptance.json)
use identical workload settings for both retained arms. The full profile ran
before the default profile. To reproduce the default arm,
use `vllm-mach-serve --model MODEL --kv-cache-memory-bytes 8218214400
--no-gdn-persistent --no-gdn-ba-overlap
--no-lossless-prefill --no-owner-prefill` without the optional acceleration flags.
The opt-outs are needed to reproduce these September 15 measurements with
the current launcher.

To reproduce the scored workload against either service:

```bash
python tools/benchmark_native_mxfp6.py \
  --base-url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL_NAME \
  --num-prompts 64 --input-tokens 3000 --output-tokens 1000 \
  --max-concurrency 32 --warmup-requests 32 --warmup-output-tokens 128 \
  --contract-seed 20260915 --request-seed-base 2026091500 \
  --json-out result.json
```

Start the optimized service with the full command in the installation guide.
For the historical default profile, use the opt-outs above and keep
the same KV allocation.

The native extension wheels used for acceptance matched the existing compiler
and ABI contracts; this turn did not rebuild them. Docker was unavailable in
the validation environment, so the revised Dockerfile was not built here.

Qwen3.5-35B-A3B uses the separate [MoE TP2 integration](qwen35-moe.md),
with adapters imported from `mxfp6_sm120` revision `cd4e964c391fcb8aaf1a27d28a63d778e3a38ece`.
