# SPDX-License-Identifier: Apache-2.0
"""Self-contained six-point streaming benchmark for an already-running profile.

Run with ``python -m vllm_mach.mxfp8.benchmark``. Never starts/stops a service.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

POINTS = (4,16,24,32,48,64)
REQUEST_COUNTS = {4:40,16:160,24:120,32:160,48:240,64:320}
INPUT_TOKENS = 3000
OUTPUT_TOKENS = 1000
CONTRACT_SEED = 20260924
REQUEST_SEED_BASE = 2026092400
PREWARM_CONTRACT_SEED = 20261003
PREWARM_REQUEST_SEED_BASE = 2026100300


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)

def make_contract(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[int]]]:
    rng = random.Random(args.contract_seed)
    if args.prompt_manifest:
        manifest = json.loads(args.prompt_manifest.read_text())
        assert len(manifest["prompts"]) >= args.num_prompts
        prompts = [
            (row["token_ids"] * math.ceil(args.input_tokens / len(row["token_ids"])))[
                : args.input_tokens
            ]
            for row in manifest["prompts"][: args.num_prompts]
        ]
    else:
        prompts = [
            [
                rng.randrange(args.token_id_low, args.token_id_high)
                for _ in range(args.input_tokens)
            ]
            for _ in range(args.num_prompts)
        ]
    offsets = [0.0]
    arrival_rng = random.Random(args.contract_seed)
    for _ in range(1, args.num_prompts):
        offsets.append(
            offsets[-1] + arrival_rng.expovariate(args.request_rate)
            if args.request_rate
            else 0.0
        )
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
            "top_k": args.top_k,
            "top_p": args.top_p,
            "presence_penalty": 0.0,
            "ignore_eos": True,
            "per_request_seed_base": args.request_seed_base,
        },
        "arrival_offsets_s": offsets,
        "prompt_source": "frozen ShareGPT repeated tokens"
        if args.prompt_manifest
        else "uniform token IDs",
    }
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
    *,
    top_k: int = -1,
    top_p: float = 1.0,
    arrival_offset: float = 0.0,
) -> dict[str, Any]:
    if arrival_offset:
        await asyncio.sleep(arrival_offset)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_k": top_k,
        "top_p": top_p,
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
                    decoder = codecs.getincrementaldecoder("utf-8")()
                    async for chunk in response.content.iter_any():
                        buffer += decoder.decode(chunk)
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
                                if choices and (
                                    choices[0].get("text")
                                    or (
                                        output_tokens == 1
                                        and choices[0].get("finish_reason") is not None
                                        and first_token_at is None
                                    )
                                ):
                                    now = time.perf_counter()
                                    if first_token_at is None:
                                        first_token_at = now
                                    token_event_times.append(now)
                                    text_parts.append(choices[0].get("text") or "")
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
        else None
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
        "response_text": text,
    }


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def protocol_definition():
    return {
        "version":1, "points":list(POINTS),
        "scored_requests":{str(c):REQUEST_COUNTS[c] for c in POINTS},
        "input_tokens":INPUT_TOKENS,"output_tokens":OUTPUT_TOKENS,
        "prompt_source":"uniform token IDs", "token_id_interval":[1000,240000],
        "contract_seed":CONTRACT_SEED,"request_seed_base":REQUEST_SEED_BASE,
        "sampling":{"temperature":0.0,"top_k":-1,"top_p":1.0,
                    "presence_penalty":0.0,"ignore_eos":True},
        "arrival_offset_s":0.0,
        "prewarm":{"requests":"c","input_tokens":3000,"output_tokens":256,
                   "contract_seed":PREWARM_CONTRACT_SEED,
                   "request_seed_base":PREWARM_REQUEST_SEED_BASE,"internal_warmup_requests":0},
        "screen_internal_warmup":{"requests":"c","output_tokens":128,
                                  "prompt_selection":"first c scored prompts",
                                  "request_seed_offset":100000},
        "stream":True,"stream_options":{"include_usage":True},
        "duration":"perf_counter around gather of scored requests only",
        "ttft":"first text-bearing event minus semaphore-admitted request start",
        "tpot":"(response end minus first token)/(completion_tokens-1)",
        "throughput":"successful completion token usage / scored duration",
        "percentile":"linear interpolation at (count-1)*quantile",
    }


def phase_args(base_url, model, concurrency, phase, *, timeout_s=7200.0):
    if concurrency not in POINTS or phase not in ("prewarm","screen"):
        raise ValueError("supported six-point concurrency and phase required")
    prewarm = phase == "prewarm"
    return argparse.Namespace(
        base_url=base_url,model=model,max_concurrency=concurrency,
        num_prompts=concurrency if prewarm else REQUEST_COUNTS[concurrency],
        input_tokens=INPUT_TOKENS,output_tokens=256 if prewarm else OUTPUT_TOKENS,
        contract_seed=PREWARM_CONTRACT_SEED if prewarm else CONTRACT_SEED,
        request_seed_base=PREWARM_REQUEST_SEED_BASE if prewarm else REQUEST_SEED_BASE,
        token_id_low=1000,token_id_high=240000,prompt_manifest=None,request_rate=0,
        top_k=-1,top_p=1.0,warmup_requests=0 if prewarm else concurrency,
        warmup_output_tokens=32 if prewarm else 128,timeout_s=timeout_s)


def aggregate_requests(results, duration):
    successful = [item for item in results if item["success"]]
    prompt_tokens = sum(int(item["usage"]["prompt_tokens"]) for item in successful)
    completion_tokens = sum(
        int(item["usage"]["completion_tokens"]) for item in successful
    )
    ttfts = [item["ttft_s"] * 1000 for item in successful]
    tpots = [item["tpot_s"] * 1000 for item in successful if item["tpot_s"] is not None]
    itls = [interval * 1000 for item in successful for interval in item["itl_s"]]
    aggregate = {
        "completed": len(successful),
        "requested": len(results),
        "duration_s": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_throughput_per_s": len(successful) / duration if duration else None,
        "output_throughput_tokens_per_s": completion_tokens / duration if duration else None,
        "total_throughput_tokens_per_s": (prompt_tokens + completion_tokens) / duration if duration else None,
        "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else None,
        "p99_ttft_ms": percentile(ttfts, 0.99),
        "mean_tpot_ms": statistics.fmean(tpots) if tpots else None,
        "p99_tpot_ms": percentile(tpots, 0.99),
        "mean_itl_ms": statistics.fmean(itls) if itls else None,
        "p99_itl_ms": percentile(itls, 0.99),
    }
    latencies = [item["latency_s"] * 1000 for item in successful]
    aggregate.update(mean_latency_ms=statistics.fmean(latencies) if latencies else None,
                     p99_latency_ms=percentile(latencies,0.99))
    return aggregate


async def run_phase(args):
    """Original streaming request/timing path with auditable warmup results."""
    import aiohttp
    contract, prompts = make_contract(args)
    identity = {"contract_sha256":canonical_hash(contract),
                "prompt_token_ids_sha256":canonical_hash(prompts)}
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 16))
    semaphore = asyncio.Semaphore(args.max_concurrency)
    url = args.base_url.rstrip("/") + "/v1/completions"
    warmup_results = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmups = [
            request_one(
                session,semaphore,url,args.model,prompt,
                min(args.warmup_output_tokens,args.output_tokens),
                args.request_seed_base + 100000 + index,index,
                top_k=args.top_k,top_p=args.top_p,
            ) for index,prompt in enumerate(prompts[:args.warmup_requests])
        ]
        if warmups:
            warmup_results = await asyncio.gather(*warmups)
        for item in warmup_results:
            item.update(seed=args.request_seed_base+100000+item["request_index"],
                        excluded_from_scoring=True)
        if warmup_results and not all(item["success"] for item in warmup_results):
            return {"schema_version":2,"model":args.model,"contract":contract,**identity,
                    "benchmark_started_epoch":None,"aggregate":aggregate_requests([],0.0),
                    "requests":[],"aborted_before_scoring":True,
                    "error":"warmup request failed",
                    "warmup":{"requests":len(warmup_results),
                              "output_tokens":min(args.warmup_output_tokens,args.output_tokens),
                              "results":warmup_results}}
        benchmark_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                request_one(
                    session,semaphore,url,args.model,prompt,args.output_tokens,
                    args.request_seed_base + index,index,
                    top_k=args.top_k,top_p=args.top_p,
                    arrival_offset=contract["arrival_offsets_s"][index],
                ) for index,prompt in enumerate(prompts)
            ]
        )
        duration = time.perf_counter() - benchmark_start
    for item in results:
        item.update(seed=args.request_seed_base+item["request_index"],
                    prompt_token_ids_sha256=canonical_hash(prompts[item["request_index"]]))
    return {
        "schema_version":2,"model":args.model,"contract":contract,**identity,
        "benchmark_started_epoch":time.time()-(time.perf_counter()-benchmark_start),
        "aggregate":aggregate_requests(results,duration),"requests":results,
        "warmup":{"requests":min(args.warmup_requests,len(prompts)),
                  "output_tokens":min(args.warmup_output_tokens,args.output_tokens),
                  "results":warmup_results},
    }


def json_safe(value):
    """Keep failed-request timing artifacts valid JSON without changing metrics."""
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [json_safe(item) for item in value]
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    return value


def write_json(path, value):
    text = json.dumps(json_safe(value),indent=2,allow_nan=False)+"\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def phase_complete(result, args):
    aggregate = result["aggregate"]
    return (not result.get("aborted_before_scoring")
            and aggregate["requested"] == aggregate["completed"] == args.num_prompts
            and aggregate["prompt_tokens"] == args.num_prompts*args.input_tokens
            and aggregate["completion_tokens"] == args.num_prompts*args.output_tokens)


def summarize_points(points, selected_points):
    """Pool scored requests over the sum of measured point durations only."""
    results = [request for point in points for request in point["screen"]["requests"]]
    duration = sum(point["screen"]["aggregate"]["duration_s"] for point in points)
    aggregate = aggregate_requests(results,duration)
    full_selection = tuple(selected_points) == POINTS
    full_points = tuple(point["concurrency"] for point in points) == POINTS
    complete = (full_selection and full_points and all(point["success"] for point in points)
                and aggregate["requested"] == aggregate["completed"] == 1040
                and aggregate["prompt_tokens"] == 3120000
                and aggregate["completion_tokens"] == 1040000)
    return {"complete":complete,"all_selected_points_succeeded":
                len(points)==len(selected_points) and all(point["success"] for point in points),
            "selection":"sixpoint" if full_selection else "smoke_subset",
            "aggregate":aggregate,"duration_excludes":"prewarm, internal warmup, HTTP session setup, file writes",
            "note":"Pooled throughput is sum of output tokens / sum of measured point durations; compare individual points for champion performance."}


async def run_suite(args):
    out = args.outdir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    protocol = protocol_definition()
    summary = {"schema_version":1,"model":args.model,"base_url":args.base_url,
               "protocol":protocol,"protocol_sha256":canonical_hash(protocol),
               "selected_points":list(args.points),"points":[],"complete":False}
    records = []
    for c in args.points:
        prewarm_args = phase_args(args.base_url,args.model,c,"prewarm",timeout_s=args.timeout_s)
        prewarm = await run_phase(prewarm_args)
        prewarm_file = out/f"c{c}_prewarm.json"
        prewarm_sha = write_json(prewarm_file,prewarm)
        if not phase_complete(prewarm,prewarm_args):
            summary.update(error=f"c{c} prewarm failed",failed_phase="prewarm",failed_concurrency=c)
            summary["failed_phase_file"] = prewarm_file.name
            summary["failed_phase_sha256"] = prewarm_sha
            break
        screen_args = phase_args(args.base_url,args.model,c,"screen",timeout_s=args.timeout_s)
        screen = await run_phase(screen_args)
        screen_file = out/f"c{c}_screen.json"
        screen_sha = write_json(screen_file,screen)
        success = phase_complete(screen,screen_args)
        point = {"concurrency":c,"success":success,"prewarm":prewarm,"screen":screen}
        records.append(point)
        summary["points"].append({"concurrency":c,"success":success,
            "prewarm":{"file":prewarm_file.name,"sha256":prewarm_sha,
                       "aggregate":prewarm["aggregate"],"contract_sha256":prewarm["contract_sha256"]},
            "screen":{"file":screen_file.name,"sha256":screen_sha,
                      "aggregate":screen["aggregate"],"contract_sha256":screen["contract_sha256"],
                      "prompt_token_ids_sha256":screen["prompt_token_ids_sha256"]}})
        print(json.dumps({"concurrency":c,**screen["aggregate"]},allow_nan=False),flush=True)
        if not success:
            summary.update(error=f"c{c} scored/internal warmup phase failed",failed_phase="screen",failed_concurrency=c)
            break
    summary.update(summarize_points(records,args.points))
    if summary.get("error"):summary["all_selected_points_succeeded"] = False
    write_json(out/"sixpoint.json",summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",required=True,help="root URL of the already-running profile service")
    parser.add_argument("--model",required=True)
    parser.add_argument("--outdir",required=True,type=Path,help="new directory for raw requests and sixpoint.json")
    parser.add_argument("--points",nargs="+",type=int,choices=POINTS,default=list(POINTS))
    parser.add_argument("--timeout-s",type=float,default=7200.0)
    args = parser.parse_args(argv)
    if tuple(args.points) != tuple(c for c in POINTS if c in args.points):
        parser.error("points must be unique and in canonical six-point order")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be positive finite")
    if args.outdir.exists():parser.error("outdir must be new")
    return args


def main():
    result = asyncio.run(run_suite(parse_args()))
    print(json.dumps({"complete":result["complete"],"selection":result["selection"],
                      "aggregate":result["aggregate"]},indent=2,allow_nan=False))
    if not result["all_selected_points_succeeded"]:raise SystemExit(2)


if __name__ == "__main__":main()
