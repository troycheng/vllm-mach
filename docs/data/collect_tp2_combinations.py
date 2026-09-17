"""Keep full/default contracts and forward/reverse blocks separate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_native_comparison import bootstrap
import numpy as np

ROWS = (4, 16, 24, 32)
VARIANTS = ('control', 'empty', 'gate', 'both')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(contract, profile, empty, gate):
    full = profile == 'full'
    assert contract['precision_profile'] == profile
    assert contract['config']['mamba_ssm_cache_dtype'] == ('float16' if full else 'float32')
    assert contract['config']['dtype'] == 'bfloat16'
    assert contract['repeats'] == 5 and not contract['profile']
    assert contract['gdn_empty_output'] == str(int(empty))
    assert contract['fused_attention_quant'] == str(int(gate))
    assert contract['gdn_recurrent_tile'] == '32' and contract['strided_gdn_ba'] == '0'
    for name in ('VLLM_QWEN3_5_FP16_SSM', 'VLLM_SM120_LOSSLESS_PREFILL',
                 'VLLM_SM120_OWNER_PREFILL', 'VLLM_HYBRID_NVFP4_LM_HEAD'):
        assert contract['environment'][name] == str(int(full)), name
    assert contract['environment']['VLLM_MACH_GDN_PERSISTENT'] == '1'
    assert contract['environment']['VLLM_MACH_GDN_BA_OVERLAP'] == '1'


def validate_fidelity_contract(contract, baseline, profile, variant, rows):
    assert contract['arm'] == ('full_gdn' if profile == 'full' else 'gdn')
    assert contract['physical_rows'] == rows
    for key in ('extension_library_sha256', 'fused_gdn_quant', 'strided_gdn_ba',
                'gdn_recurrent_tile', 'fused_swiglu_quant', 'environment'):
        assert contract[key] == baseline[key], key
    for key in ('model', 'dtype', 'tensor_parallel_size', 'mamba_ssm_cache_dtype'):
        assert contract['llm_args'][key] == baseline['config'][key], key
    assert contract['gdn_empty_output'] == str(int(variant == 'both'))
    assert contract['fused_attention_quant'] == str(int(variant == 'both'))
    assert contract['llm_args']['logprobs_mode'] == 'raw_logprobs'


def collect_serving(serving, environment, empty, gate, serving_contracts):
    launch = read(serving / 'launch.json')
    env = launch['environment']
    assert env['VLLM_MACH_GDN_EMPTY_OUTPUT'] == str(int(empty))
    assert env['VLLM_MACH_FUSED_ATTN_QUANT'] == str(int(gate))
    for key, value in environment.items():
        assert env[key] == value, key
    points = []
    for m in ROWS:
        path = serving / f'c{m}.json'
        item = read(path)
        assert item['contract']['input_tokens'] == 3000
        assert item['contract']['output_tokens'] == 1000
        assert item['contract']['num_prompts'] == 5*m
        assert item['contract']['max_concurrency'] == m
        if m in serving_contracts:
            assert item['contract'] == serving_contracts[m]
        else:
            serving_contracts[m] = item['contract']
        assert item['warmup'] == {'requests': min(32, 5*m), 'output_tokens': 128}
        stats = item['aggregate']
        assert stats['completed'] == stats['requested'] == 5 * m
        assert stats['prompt_tokens'] == 5 * m * 3000
        assert stats['completion_tokens'] == 5 * m * 1000
        assert all(r['success'] for r in item['requests'])
        points.append(dict(concurrency=m, aggregate=stats, contract=item['contract'], sha256=sha(path)))
    return dict(points=points, launch=launch)


def collect(root, profiles=('default', 'full')):
    output = {'schema': 'mach-tp2-combinations/v1', 'profiles': {}}
    provenance = root / 'provenance.json'
    if provenance.exists():
        output['provenance'] = read(provenance)
        output['provenance_sha256'] = sha(provenance)
    serving_contracts = {}
    for profile in profiles:
        full = profile == 'full'
        baseline = read(root / f'{profile}-r0-control/contract.json')
        data = dict(contract=baseline, decode={}, serving={}, fidelity={})
        expected_hashes = {}
        for variant in VARIANTS:
            empty, gate = variant in ('empty', 'both'), variant in ('gate', 'both')
            blocks = []
            for block in (0, 1):
                directory = root / f'{profile}-r{block}-{variant}'
                contract = read(directory / 'contract.json')
                validate_contract(contract, profile, empty, gate)
                for key in baseline:
                    if key not in ('gdn_empty_output', 'fused_attention_quant'):
                        assert contract[key] == baseline[key], (profile, variant, key)
                summary = read(directory / 'summary.json')
                assert [r['rows'] for r in summary] == (list(ROWS) if block == 0 else list(reversed(ROWS)))
                rows = []
                for m in ROWS:
                    rates, itls, prefill, hashes, raw = [], [], [], [], []
                    for trial in range(5):
                        path = directory / f'm{m}-trial{trial}/result.json'
                        result = read(path)
                        assert len(result['metrics']) == m
                        assert all(v['num_generation_tokens'] == 1025 and not v['is_corrupted']
                                   for v in result['metrics'])
                        assert {r['rank'] for r in result['ranks']} == {0, 1}
                        for rank in result['ranks']:
                            assert rank['empty_output_layers'] == 48 * empty
                            assert rank['fused_attention_layers'] == 16 * gate
                            assert rank['fused_gdn_quant_layers'] == 48
                            assert rank['fused_swiglu_layers'] == 64
                            assert rank['recurrent_tile8_layers'] == 0
                            assert rank['hybrid_head_modules'] == int(full)
                            assert rank['gdn_state_dtypes'] == ['torch.float16' if full else 'torch.float32']
                            assert any(b['logical_rows'] == b['padded_rows'] == m and not b['has_prefill']
                                       for b in rank['batches'])
                        rates.append(result['output_tokens_per_s'])
                        itls.append(statistics.mean(1000*(v['last_token_ts']-v['first_token_ts'])/1024
                                                   for v in result['metrics']))
                        prefill.append(statistics.mean(1000*(v['first_token_ts']-v['scheduled_ts'])
                                                      for v in result['metrics']))
                        hashes.append(result['token_ids_sha256'])
                        raw.append(sha(path))
                    if variant == 'control' and block == 0:
                        expected_hashes[m] = hashes[0]
                    rows.append(dict(rows=m, rates=rates, mean=statistics.mean(rates),
                        sd=statistics.stdev(rates), mean_itl_ms=statistics.mean(itls),
                        mean_prefill_ms=statistics.mean(prefill), raw_sha256=raw,
                        token_hashes=hashes,
                        identical_to_control=all(h == expected_hashes[m] for h in hashes)))
                blocks.append(dict(block=block, rows=rows, contract_sha256=sha(directory / 'contract.json')))
            data['decode'][variant] = blocks
            arm = 'full_gdn' if full else 'gdn'
            serving = root / f'{profile}-serving-{variant}' / arm
            data['serving'][variant] = collect_serving(serving, baseline['environment'], empty, gate, serving_contracts)
            extra = root / f'{profile}-serving-repeat-{variant}' / arm
            if full and variant in ('control', 'both') and extra.exists():
                marker = root / f'{profile}-serving-repeat-{variant}.complete.json'
                assert marker.exists(), f'Incomplete serving repeat: {extra}'
                data['serving'][variant]['repeat'] = collect_serving(extra, baseline['environment'], empty, gate, serving_contracts)
                data['serving'][variant]['repeat']['completion_sha256'] = sha(marker)
        for m, reference in [(4, 'gdn-m4-fidelity.json'), (32, 'native-fidelity.json')]:
            bf16 = read(Path(__file__).with_name(reference))['queries']
            pair = {}
            baseline_records = read(root / f'{profile}-fidelity-control-m{m}/records.json')
            for variant in ('control', 'both'):
                directory = root / f'{profile}-fidelity-{variant}-m{m}'
                assert read(directory / 'COMPLETE.json')['target_tokens'] == 10479
                records = read(directory / 'records.json')
                contract = read(directory / 'contract.json')
                validate_fidelity_contract(contract, baseline, profile, variant, m)
                gdn = read(directory / 'gdn.json')
                assert {r['rank'] for r in gdn} == {0, 1}
                assert all(r['prepared_layers'] == r['fused_output_layers'] == 48 for r in gdn)
                assert [r['id'] for r in records] == [q['id'] for q in bf16]
                assert [r['id'] for r in records] == [q['id'] for q in baseline_records]
                errors = [float(np.abs(np.array(r['gold_logprobs']) - q['bf16_logprobs']).mean())
                          for r, q in zip(records, bf16, strict=True)]
                differences = [abs(x-y) for r, q in zip(records, baseline_records, strict=True)
                    for x, y in zip(r['gold_logprobs'], q['gold_logprobs'], strict=True)]
                assert len(differences) == 10479
                pair[variant] = dict(mae=float(np.mean(errors)), ci95=bootstrap(errors),
                    max_difference_from_control=max(differences), repeat=read(directory / 'repeat.json'),
                    records_sha256=sha(directory / 'records.json'), contract=contract, gdn=gdn)
                if full:
                    pair[variant]['head'] = read(directory / 'head.json')
            data['fidelity'][str(m)] = pair
        output['profiles'][profile] = data
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--profiles', choices=('default', 'full'), nargs='+', default=['default', 'full'])
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('tp2-combinations.json'))
    a = p.parse_args()
    a.output.write_text(json.dumps(collect(a.root, a.profiles), indent=2) + '\n')


if __name__ == '__main__':
    main()
