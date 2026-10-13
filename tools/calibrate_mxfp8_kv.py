#!/usr/bin/env python3
"""Freeze FP8 KV scales from seeded random tokens, without evaluation inputs."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import struct

LAYERS = tuple(range(3, 32, 4))


def install_observer(wrapper):
    import torch
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    worker = getattr(wrapper, 'worker', wrapper)
    assert worker.model_config.enforce_eager
    original = FlashAttentionImpl.do_kv_cache_update
    assert tuple(inspect.signature(original).parameters) == (
        'self', 'layer', 'key', 'value', 'kv_cache', 'slot_mapping')
    assert not hasattr(worker, '_random_kv_observer')
    state = {'active': False, 'stats': {}}
    worker._random_kv_observer = state

    def observed(self, layer, key, value, kv_cache, slot_mapping):
        if state['active']:
            index = int(layer.layer_name.split('.layers.')[1].split('.')[0])
            assert index in LAYERS and layer.kv_cache_torch_dtype == torch.bfloat16
            assert not torch.cuda.is_current_stream_capturing()
            if slot_mapping is not None:
                valid = slot_mapping >= 0
                for limb, tensor in (('k', key), ('v', value)):
                    x = tensor[:slot_mapping.numel()][valid].float()
                    if x.numel():
                        name = f'{index}.{limb}'
                        stat = state['stats'].setdefault(name, {
                            'amax': torch.zeros((), device=worker.device),
                            'nonfinite': torch.zeros((), device=worker.device, dtype=torch.int64),
                            'elements': 0})
                        stat['amax'] = torch.maximum(stat['amax'], x.abs().max())
                        stat['nonfinite'] += (~torch.isfinite(x)).sum()
                        stat['elements'] += x.numel()
        return original(self, layer, key, value, kv_cache, slot_mapping)

    FlashAttentionImpl.do_kv_cache_update = observed
    return {'boundary': 'post-RoPE BF16 K/V cache write, valid slots only',
            'backend': 'FLASH_ATTN', 'eager': True}


def reset_observer(wrapper):
    worker = getattr(wrapper, 'worker', wrapper)
    worker._random_kv_observer.update(active=True, stats={})


def snapshot_observer(wrapper):
    worker = getattr(wrapper, 'worker', wrapper)
    state = worker._random_kv_observer
    state['active'] = False
    return {name: {k: value if isinstance(value, int) else value.item()
                   for k, value in stat.items()} for name, stat in state['stats'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[931007, 931021, 931043])
    parser.add_argument('--requests', type=int, default=192)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--lengths', type=Path,
                        help='Optional JSON list of calibration-only input lengths; no token contents')
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.tokens <= 4096:
        parser.error('requests must be positive and tokens within 1..4096')
    lengths = json.loads(args.lengths.read_text()) if args.lengths else [args.tokens] * args.requests
    if (not isinstance(lengths, list) or len(lengths) != args.requests
            or any(type(n) is not int or not 1 <= n <= args.tokens for n in lengths)):
        parser.error('lengths must contain requests integers within 1..tokens')
    args.outdir.mkdir(parents=True, exist_ok=False)
    # No complete-profile hooks or model-sidecar KV scale installation.
    from vllm_mach.mxfp8.install import load_manifest
    manifest = load_manifest()
    os.environ.pop('VLLM_MACH_PROFILE', None)
    os.environ.update(VLLM_PLUGINS='', VLLM_USE_V2_MODEL_RUNNER='1',
                      VLLM_ALLOW_INSECURE_SERIALIZATION='1', VLLM_FLASH_ATTN_VERSION='2',
                      OMP_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1',
                      **manifest['required_disabled_legacy_environment'],
                      **manifest['required_disabled_compact_environment'])
    for env, name in [('VLLM_CACHE_ROOT', 'vllm'), ('TRITON_CACHE_DIR', 'triton'),
                      ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        os.environ[env] = str(args.outdir/'cache'/name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special = set(tokenizer.all_special_ids)
    vocabulary = sorted(set(tokenizer.get_vocab().values()) - special)
    # Freeze all inputs before model execution; never read any evaluation file.
    corpora = {}
    for seed in args.seeds:
        rng = random.Random(seed)
        rows = [[rng.choice(vocabulary) for _ in range(n)] for n in lengths]
        payload = json.dumps(rows, separators=(',', ':')).encode()
        (args.outdir/f'inputs-{seed}.json').write_bytes(payload+b'\n')
        corpora[seed] = (rows, hashlib.sha256(payload).hexdigest())
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    torch.set_num_threads(1)
    engine = LLM(model=args.model, tokenizer=args.model, dtype='bfloat16',
                 tensor_parallel_size=1, language_model_only=True,
                 attention_backend='FLASH_ATTN', kv_cache_dtype='bfloat16',
                 mamba_ssm_cache_dtype='float32', enforce_eager=True,
                 enable_prefix_caching=False, enable_chunked_prefill=True,
                 max_num_seqs=32, max_model_len=8192, max_num_batched_tokens=2048,
                 kv_cache_memory_bytes=4*2**30, seed=20260907,
                 quantization='compressed-tensors',
                 kernel_config={'linear_backend': 'flashinfer_cutlass'})
    try:
        installation = engine.collective_rpc(install_observer)
        for seed, (corpus, digest) in corpora.items():
            engine.collective_rpc(reset_observer)
            for start in range(0, len(corpus), 32):
                prompts = [TokensPrompt(prompt_token_ids=x) for x in corpus[start:start+32]]
                outputs = engine.generate(prompts, SamplingParams(
                    temperature=0, max_tokens=1, ignore_eos=True, detokenize=False), use_tqdm=False)
                assert len(outputs) == len(prompts)
            stats = engine.collective_rpc(snapshot_observer)[0]
            assert set(stats) == {f'{i}.{limb}' for i in LAYERS for limb in ('k', 'v')}
            scales = []
            for index in LAYERS:
                row = {'layer_index': index, 'module_name': f'model.layers.{index}.self_attn.attn'}
                for limb in ('k', 'v'):
                    stat = stats[f'{index}.{limb}']
                    assert stat['nonfinite'] == 0 and stat['elements'] > 0
                    value = 1.1 * stat['amax'] / 448 if stat['amax'] else 1.0
                    row[f'{limb}_scale'] = struct.unpack('f', struct.pack('f', value))[0]
                scales.append(row)
            bank = dict(profile='qwen35-4b-mxfp8-champion-v1', kv_scales=scales,
                        calibration=dict(seed=seed, requests=args.requests, max_tokens=args.tokens,
                            total_input_tokens=sum(lengths),
                            input_lengths_sha256=hashlib.sha256(json.dumps(lengths,
                                separators=(',', ':')).encode()).hexdigest(),
                            input_sha256=digest, formula='float32(1.1 * random_absmax / 448)',
                            evaluation_inputs_read=False, source='uniform non-special tokenizer IDs',
                            model=args.model, observer=installation, stats=stats))
            (args.outdir/f'scales-{seed}.json').write_text(json.dumps(bank, indent=2)+'\n')
            print('FROZEN_RANDOM_KV_SCALES', seed, flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
