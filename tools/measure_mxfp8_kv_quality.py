#!/usr/bin/env python3
"""Measure fixed-M teacher logprobs for a frozen 4B KV ablation manifest.

The same manifest must be used in a separate BF16-KV reference engine. This
measures numerical fidelity, not task correctness. No calibration is performed.
"""
import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def install_teacher(wrapper, rows):
    import teacher_decode
    worker = getattr(wrapper, 'worker', wrapper)
    receipt = teacher_decode.install(worker, native=True)
    worker._k4o_teacher_shape_state['physical_rows'] = rows
    return {**receipt, 'physical_rows': rows}


def inspect_worker(wrapper):
    from vllm_mach.mxfp8.profile import write_receipt
    return write_receipt(getattr(wrapper, 'worker', wrapper), 'kv_quality')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def score(llm, samples, rows, budget):
    import teacher_decode
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    records, first = [], None
    cohorts = len(samples) // rows
    for index in range(cohorts + 1):
        batch = samples[(index % cohorts) * rows:(index % cohorts + 1) * rows]
        max_gold = max(len(x['target_token_ids']) for x in batch)
        if sum(len(x['prompt_token_ids']) - 1 for x in batch) > budget:
            raise ValueError('Whole cohort prefill must fit in one batch')
        prompts, params, sequences = [], [], []
        for sample in batch:
            p, t = sample['prompt_token_ids'], sample['target_token_ids']
            seq = [p[-1]] + t + [t[-1]] * (max_gold - len(t))
            if len(p) - 1 + len(seq) > 768:
                raise ValueError('Short-gold manifest exceeds 768 tokens')
            prompts.append(TokensPrompt(prompt_token_ids=p[:-1]))
            sequences.append(seq)
            params.append(SamplingParams(temperature=0, max_tokens=len(seq),
                ignore_eos=True, logprobs=1, detokenize=False,
                extra_args={teacher_decode.KEY: {'id': sample['id'],
                    'forced_ids': seq, 'pad_id': seq[-1]}}))
        core = llm.llm_engine.engine_core
        core.call_utility('pause_scheduler', 'keep', False)
        try:
            if not core.call_utility('is_scheduler_paused'):
                raise RuntimeError('Scheduler did not pause')
            internal = llm.enqueue(prompts, params, use_tqdm=False)
            states = llm.llm_engine.output_processor.request_states
            external = [str(states[str(i)].external_req_id) for i in internal]
        finally:
            core.call_utility('resume_scheduler')
        outputs = {str(r.request_id): r for r in llm.wait_for_completion(use_tqdm=False)}
        if set(outputs) != set(external):
            raise RuntimeError('Completed request IDs differ')
        batch_records = []
        for sample, seq, key in zip(batch, sequences, external, strict=True):
            generation = outputs[key].outputs[0]
            if list(generation.token_ids) != seq or generation.finish_reason != 'length':
                raise RuntimeError('Teacher output differs')
            values = [float(generation.logprobs[j + 1][t].logprob)
                      for j, t in enumerate(sample['target_token_ids'])]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('Non-finite gold logprob')
            batch_records.append(dict(id=sample['id'], domain=sample['domain'],
                                      gold_logprobs=values))
        observations = llm.collective_rpc(teacher_decode.observations)
        for rank in observations:
            calls = [c for c in rank['calls'] if 1 <= c['offset'] <= max_gold]
            if (len(calls) != max_gold or {c['offset'] for c in calls} != set(range(1, max_gold + 1))
                    or any(c['num_tokens'] != rows or c['num_reqs'] != rows
                           or set(c['ids']) != {s['id'] for s in batch} for c in calls)):
                raise RuntimeError('Actual fixed physical decode rows/offsets differ')
        if index < cohorts:
            records.extend(batch_records)
            if index == 0:
                first = batch_records
        else:
            delta = [abs(x-y) for a,b in zip(first,batch_records,strict=True)
                     for x,y in zip(a['gold_logprobs'],b['gold_logprobs'],strict=True)]
            repeat = dict(scope='first cohort', max_abs=max(delta))
        print('KV_QUALITY_COHORT', index, flush=True)
    return records, repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--kv-cache-dtype', choices=('bfloat16', 'fp8_e4m3'), required=True)
    parser.add_argument('--kv-scales', type=Path)
    parser.add_argument('--rows', type=int, choices=(32,64), default=32)
    args = parser.parse_args()
    samples = json.loads(args.manifest.read_text())['samples']
    if (not samples or len(samples) % args.rows or len({s['id'] for s in samples}) != len(samples)
            or any(len(s['prompt_token_ids']) < 2 or not s['target_token_ids']
                   or any(type(t) is not int or t < 0 for t in s['prompt_token_ids'] + s['target_token_ids'])
                   for s in samples)):
        parser.error('Manifest needs unique IDs, nonempty token sequences and complete cohorts')
    args.outdir.mkdir(parents=True, exist_ok=False)
    diagnostics = Path(__file__).resolve().parent / 'quantization'
    sys.path.insert(0, str(diagnostics))
    os.environ['PYTHONPATH'] = os.pathsep.join((str(diagnostics), os.environ.get('PYTHONPATH', '')))
    os.environ.update(K4O_QUALITY_ARM='m32', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                      OMP_NUM_THREADS='1', VLLM_SMALL_BATCH_ATTN='0')
    from vllm_mach.mxfp8.serve import configure_environment
    from vllm_mach.mxfp8.profile import quality_contract, check_environment, check_runtime_sources
    from vllm_mach.mxfp8.prepare_model import validate_model
    configure_environment(args.outdir/'run', quality_rows=args.rows,
                          kv_cache_dtype=args.kv_cache_dtype, kv_scales=args.kv_scales)
    kwargs = dict(model=str(args.model.resolve()), tokenizer=str(args.model.resolve()),
                  **quality_contract(args.rows))
    write(args.outdir/'contract.json', dict(llm_args=kwargs,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        teacher_sha256=hashlib.sha256((diagnostics/'teacher_decode.py').read_bytes()).hexdigest(),
        model=validate_model(args.model, verify_weights=True), versions=check_environment(),
        runtime_sources=check_runtime_sources()))
    import torch
    from vllm import LLM
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = LLM(**kwargs)
    try:
        write(args.outdir/'runtime.json', engine.collective_rpc(inspect_worker))
        write(args.outdir/'teacher.json', engine.collective_rpc(
            functools.partial(install_teacher, rows=args.rows)))
        records, repeat = score(engine, samples, args.rows, kwargs['max_num_batched_tokens'])
        write(args.outdir/'records.json', records)
        write(args.outdir/'complete.json', dict(queries=len(records),
            gold_tokens=sum(len(x['gold_logprobs']) for x in records), repeat=repeat))
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
