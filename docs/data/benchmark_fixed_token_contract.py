#!/usr/bin/env python3
"""Deterministic OpenAI-completions benchmark with response fingerprints."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def make_contract(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[int]]]:
    rng = random.Random(args.contract_seed)
    prompts = [
        [rng.randrange(args.token_id_low, args.token_id_high) for _ in range(args.input_tokens)]
        for _ in range(args.num_prompts)
    ]
    contract = {
        "version": 1,
        "num_prompts": args.num_prompts,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "token_id_low": args.token_id_low,
        "token_id_high": args.token_id_high,
        "contract_seed": args.contract_seed,
        "max_concurrency": args.max_concurrency,
        "sampling": {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "ignore_eos": True,
            "per_request_seed_base": args.request_seed_base,
        },
        "prompt_sha256": [
            hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            for prompt in prompts
        ],
        "arrival_offsets_s": [0.0] * args.num_prompts,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract, prompts


async def request_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    seed: int,
    request_index: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    queued_epoch = time.time()
    queued_at = time.perf_counter()
    async with semaphore:
        started_at = time.perf_counter()
        first_token_at: float | None = None
        token_event_times: list[float] = []
        text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        error = ""
        status = 0
        try:
            async with session.post(url, json=payload) as response:
                status = response.status
                if status != 200:
                    error = await response.text()
                else:
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode("utf-8")
                        while "\n\n" in buffer:
                            event, buffer = buffer.split("\n\n", 1)
                            for line in event.splitlines():
                                if not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                if not raw or raw == "[DONE]":
                                    continue
                                data = json.loads(raw)
                                if data.get("usage") is not None:
                                    usage = data["usage"]
                                choices = data.get("choices") or []
                                if choices and choices[0].get("text"):
                                    now = time.perf_counter()
                                    if first_token_at is None:
                                        first_token_at = now
                                    token_event_times.append(now)
                                    text_parts.append(choices[0]["text"])
        except Exception as exc:  # preserve failures in the evidence artifact
            error = repr(exc)
        ended_at = time.perf_counter()

    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    success = (
        status == 200
        and not error
        and usage is not None
        and prompt_tokens == len(prompt)
        and completion_tokens == output_tokens
        and first_token_at is not None
    )
    ttft = first_token_at - started_at if first_token_at is not None else math.nan
    tpot = (
        (ended_at - first_token_at) / (completion_tokens - 1)
        if first_token_at is not None and completion_tokens > 1
        else math.nan
    )
    text = "".join(text_parts)
    return {
        "request_index": request_index,
        "queued_epoch": queued_epoch,
        "started_epoch": queued_epoch + started_at - queued_at,
        "ended_epoch": queued_epoch + ended_at - queued_at,
        "success": success,
        "http_status": status,
        "error": error[:2000],
        "queue_wait_s": started_at - queued_at,
        "latency_s": ended_at - started_at,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itl_s": [b - a for a, b in zip(token_event_times, token_event_times[1:])],
        "usage": usage,
        "token_event_count": len(token_event_times),
        "response_chars": len(text),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, prompts = make_contract(args)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))
    semaphore = asyncio.Semaphore(args.max_concurrency)
    url = args.base_url.rstrip("/") + "/v1/completions"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmups = [
            request_one(
                session,
                semaphore,
                url,
                args.model,
                prompt,
                min(args.warmup_output_tokens, args.output_tokens),
                args.request_seed_base + 100000 + index,
                index,
            )
            for index, prompt in enumerate(prompts[: args.warmup_requests])
        ]
        if warmups and not all(item["success"] for item in await asyncio.gather(*warmups)):
            raise RuntimeError("warmup request failed")

        benchmark_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                request_one(
                    session,
                    semaphore,
                    url,
                    args.model,
                    prompt,
                    args.output_tokens,
                    args.request_seed_base + index,
                    index,
                )
                for index, prompt in enumerate(prompts)
            ]
        )
        duration = time.perf_counter() - benchmark_start

    successful = [item for item in results if item["success"]]
    prompt_tokens = sum(int(item["usage"]["prompt_tokens"]) for item in successful)
    completion_tokens = sum(int(item["usage"]["completion_tokens"]) for item in successful)
    ttfts = [item["ttft_s"] * 1000 for item in successful]
    tpots = [item["tpot_s"] * 1000 for item in successful]
    itls = [interval * 1000 for item in successful for interval in item["itl_s"]]
    aggregate = {
        "completed": len(successful),
        "requested": len(results),
        "duration_s": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_throughput_per_s": len(successful) / duration,
        "output_throughput_tokens_per_s": completion_tokens / duration,
        "total_throughput_tokens_per_s": (prompt_tokens + completion_tokens) / duration,
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "mean_tpot_ms": statistics.fmean(tpots),
        "p99_tpot_ms": percentile(tpots, 0.99),
        "mean_itl_ms": statistics.fmean(itls),
        "p99_itl_ms": percentile(itls, 0.99),
    }
    return {
        "schema_version": 2,
        "model": args.model,
        "contract": contract,
        "benchmark_started_epoch": time.time() - (time.perf_counter() - benchmark_start),
        "aggregate": aggregate,
        "requests": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--num-prompts", required=True, type=int)
    parser.add_argument("--input-tokens", default=1024, type=int)
    parser.add_argument("--output-tokens", default=256, type=int)
    parser.add_argument("--token-id-low", default=1000, type=int)
    parser.add_argument("--token-id-high", default=240000, type=int)
    parser.add_argument("--contract-seed", required=True, type=int)
    parser.add_argument("--request-seed-base", required=True, type=int)
    parser.add_argument("--max-concurrency", required=True, type=int)
    parser.add_argument("--warmup-requests", default=4, type=int)
    parser.add_argument("--warmup-output-tokens", default=32, type=int)
    parser.add_argument("--timeout-s", default=7200.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
    if result["aggregate"]["completed"] != result["aggregate"]["requested"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
