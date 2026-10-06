# Six-point benchmark for an installed profile

Run this client against the root URL of an already-running complete profile:

```bash
python -m vllm_mach.mxfp8.benchmark \
  --base-url http://127.0.0.1:8000 \
  --model q35-mx8-study \
  --outdir ./sixpoint-results
```

The output directory must be new. The client uses ordinary `aiohttp`, which is available in the locked serving environment. It sends completions requests only and never starts, stops, restarts or reconfigures a service. Use the model name that the existing service exposes; the benchmark does not choose a different model or deployment.

For a subset before a complete run:

```bash
python -m vllm_mach.mxfp8.benchmark \
  --base-url http://127.0.0.1:8000 \
  --model q35-mx8-study \
  --outdir ./smoke-results --points 4 16
```

Points must be unique and in the order below. A subset has `selection="smoke_subset"` and always `complete=false`. Exit code 0 means all selected points succeeded; it does not convert a subset into a complete six-point result. Failed warmup, failed HTTP/streaming requests or incomplete usage produces exit code 2 and preserves available raw evidence.

| Concurrency | Scored requests | Scored input tokens per request | Scored output tokens per request |
| ---: | ---: | ---: | ---: |
| 4 | 40 | 3000 | 1000 |
| 16 | 160 | 3000 | 1000 |
| 24 | 120 | 3000 | 1000 |
| 32 | 160 | 3000 | 1000 |
| 48 | 240 | 3000 | 1000 |
| 64 | 320 | 3000 | 1000 |

The complete run scores 1040 requests, 3,120,000 input tokens and 1,040,000 output tokens. Each point performs three phases in order:

1. Independent prewarm: `c` requests, 3000 input and 256 output tokens, `random.Random(20261003)` prompts and request seed base `2026100300`, with no internal warmup.
2. Screen internal warmup: the first `c` scored prompts, 128 output tokens and request seed `2026092400 + 100000 + index`. These requests are recorded and excluded from scoring.
3. Scored screen: the request count in the table, 3000 input and 1000 output tokens, `random.Random(20260924)` prompts and request seed `2026092400 + index`.

Each phase independently resets its Python random generator. Every prompt contains exactly the selected generator's consecutive `randrange(1000, 240000)` token IDs. All arrival offsets are zero; a semaphore limits simultaneous HTTP requests to `c`. The sampler payload is unchanged: temperature 0, top_k −1, top_p 1, presence_penalty 0, ignore_eos true, stream true and include_usage true. The per-request HTTP timeout defaults to 7200 seconds, as in the original client, and can be changed with `--timeout-s`. The original service screen's outer subprocess deadline is not part of this attached client.

The streaming parser, generator and percentile function preserve their original source. Timing also preserves the original formulas:

- Scored duration uses `perf_counter` immediately around the gather of scored requests, excluding both warmups and HTTP session setup.
- Output throughput is successful completion-token usage divided by that duration. Input and total throughput use the same denominator.
- TTFT starts when the request acquires the semaphore and ends on the first text-bearing SSE event. Queue wait is stored separately.
- TPOT is `(response end − first token time) / (completion_tokens − 1)`.
- ITL follows successive text-bearing SSE events. It is an observed streaming-event interval, not an independently reconstructed model-token timestamp.
- p99 uses linear interpolation at rank `(count − 1) × 0.99`.

Each request succeeds only with HTTP 200, no recorded error, final usage containing the exact input/output token counts and an observed first token. Failed requests remain in the raw file and are excluded from successful-token throughput and latency statistics, matching the original accounting. Additional end-to-end mean/p99 latency fields use the existing per-request latency values. Failed requests without timing observations serialize those fields as JSON null.

`c*_prewarm.json` and `c*_screen.json` contain raw response text, final usage, queue wait, latency, TTFT, TPOT, event intervals, request indices and seeds. Screen files also retain all internal warmup results. Contract and prompt hashes use SHA256 over canonical JSON; prompt IDs can be regenerated from the recorded contract. `sixpoint.json` records the fixed protocol/hash, selected points, raw-file hashes and per-point aggregates. Its pooled aggregate divides the sum of scored output tokens by the sum of measured point durations; it is not the arithmetic mean of the six throughputs. Compare the individual six points when evaluating champion performance.

`complete=true` requires all six points in their canonical order, successful prewarm/internal warmup/scored phases, and exact 1040-request/3,120,000-input/1,040,000-output totals. This flag certifies completion of the client protocol. Backend routing, model identity, calibration, graph/lifecycle correctness, no preemptions and performance equivalence still require the complete profile's separate qualification evidence.

CPU verification:

```bash
python -m unittest discover -s tests -p 'test_mxfp8_benchmark.py' -v
```

Tests cover exact request generation/phase parameters, original parser source, split UTF8 SSE payloads and timing, warmup exclusion, successful-only aggregation, pooled duration, complete/subset status and failed-request JSON. They send no network requests and need no GPU or `aiohttp` installation.
