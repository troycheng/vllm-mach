#!/usr/bin/env python3
"""Real-checkpoint TP2 P0 baseline; profiling is a separate process/run."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, nargs='+', default=[1, 4, 16, 32])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--gemm-overrides', type=Path, help='Diagnostic full-model dispatch candidates; JSON list')
    a = p.parse_args()
    if a.repeats < 5 and not a.profile:
        p.error('At least five unprofiled repeats are required')
    if any(m not in (1, 2, 4, 8, 16, 24, 32) for m in a.rows):
        p.error('Unsupported physical graph size')
    from vllm_mach.mxfp6.serve import profile_environment
    os.environ.update(profile_environment(argparse.Namespace(fp16_ssm=False, lossless_prefill=False,
        owner_prefill=False, nvfp4_lm_head=False, verify_prefill=False)))
    overrides=json.loads(a.gemm_overrides.read_text()) if a.gemm_overrides else []
    os.environ['MACH_TP2_GEMM_OVERRIDES']=json.dumps(overrides)
    os.environ['MACH_TP2_PROFILE'] = str(int(a.profile))
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.resolve()) + os.pathsep + os.environ.get('PYTHONPATH', '')
    import mxfp6
    library = Path(mxfp6.load_library())
    from vllm import LLM, SamplingParams
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(model=a.model, quantization='quark', dtype='bfloat16', tensor_parallel_size=2,
        max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=512,
        kv_cache_memory_bytes=8218214400, enable_prefix_caching=False,
        attention_backend='TRITON_ATTN', language_model_only=True, mamba_ssm_cache_dtype='float32',
        seed=20260916, disable_log_stats=False, worker_extension_cls='profile_tp2_worker.TP2Probe',
        compilation_config=dict(mode='NONE', cudagraph_mode='FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32]))
    (a.output/'contract.json').write_text(json.dumps(dict(config=config, profile=a.profile,
        repeats=a.repeats, gemm_overrides=overrides, input_tokens=2048, output_tokens={'1':129, 'other':1025},
        extension_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        fused_gdn_quant=os.environ.get('VLLM_MACH_FUSED_GDN_QUANT','auto'),
        fused_swiglu_quant=os.environ.get('VLLM_MACH_FUSED_SWIGLU_QUANT','auto'),
        packages={n:importlib.metadata.version(n) for n in ('torch','vllm','mxfp6-sm120')}, devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2))
    llm = LLM(**config)
    results = []
    for m in a.rows:
        output_tokens = 129 if m == 1 else 1025
        params = SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True, detokenize=False)
        prompts = [dict(prompt_token_ids=[1000 + (i * 37 + j * 13) % 10000 for j in range(2048)]) for i in range(m)]
        llm.generate(prompts, params, use_tqdm=False)
        trials = []
        for trial in range(1 if a.profile else a.repeats):
            trace = a.output/f'm{m}-trial{trial}'
            trace.mkdir()
            llm.collective_rpc('tp2_begin', args=(str(trace.resolve()), m))
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            duration = time.perf_counter() - start
            ranks = llm.collective_rpc('tp2_end')
            assert len(outputs) == m and all(len(o.outputs[0].token_ids) == output_tokens for o in outputs)
            metrics = [dataclasses.asdict(o.metrics) if o.metrics is not None else None for o in outputs]
            entry = dict(duration_s=duration, output_tokens_per_s=m*output_tokens/duration, metrics=metrics, ranks=ranks)
            (trace/'result.json').write_text(json.dumps(entry))
            trials.append({k:v for k,v in entry.items() if k != 'ranks'})
            print('TRIAL', m, trial, duration, flush=True)
        rates = [t['output_tokens_per_s'] for t in trials]
        results.append(dict(rows=m, trials=trials, mean_output_tokens_per_s=statistics.mean(rates),
                            stdev_output_tokens_per_s=statistics.stdev(rates) if len(rates)>1 else None))
        (a.output/'summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
