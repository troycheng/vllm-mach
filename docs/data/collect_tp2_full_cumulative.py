"""Validate full-profile P0/current ABBA measurements without mixing precision contracts."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_tp2_combinations import read, sha, validate_contract, validate_fidelity_contract
from collect_native_comparison import bootstrap

ROWS = (4, 16, 24, 32)


def collect(root):
    expected_library = {
        'p0': read(Path(__file__).with_name('tp2-p0.json'))['extension_library_sha256'],
        'current': read(Path(__file__).with_name('tp2-p2a.json'))['decode']['contract']['extension_library_sha256'],
    }
    data = {'schema': 'mach-tp2-full-cumulative/v1', 'variants': {}}
    if (root / 'provenance.json').exists():
        data['provenance'] = read(root / 'provenance.json')
    contracts = {}
    token_hashes = {}
    serving_contracts = {}
    fidelity_controls = {m: read(root / f'fidelity-p0-m{m}/records.json') for m in (4, 32)}
    for variant in ('p0', 'current'):
        accepted = variant == 'current'
        group = {'decode': [], 'serving': [], 'fidelity': {}}
        for block in (0, 1):
            label = f'decode-r{block}-{variant}'
            directory = root / label
            contract = read(directory / 'contract.json')
            validate_contract(contract, 'full', False, False)
            assert contract['extension_library_sha256'] == expected_library[variant]
            if not contracts:
                contracts.update(contract)
            for key, value in contracts.items():
                if key != 'extension_library_sha256':
                    assert contract[key] == value, key
            launch = read(root / f'{label}.complete.json')
            assert launch['library_sha256'] == expected_library[variant]
            if variant == 'p0':
                assert launch['source_commit'].startswith('637f743')
            summary = read(directory / 'summary.json')
            assert [r['rows'] for r in summary] == (list(ROWS) if block == 0 else list(reversed(ROWS)))
            rows = []
            for m in ROWS:
                rates, itls, hashes, paths = [], [], [], []
                for trial in range(5):
                    path = directory / f'm{m}-trial{trial}/result.json'
                    item = read(path)
                    assert len(item['metrics']) == m
                    assert all(v['num_generation_tokens'] == 1025 and not v['is_corrupted'] for v in item['metrics'])
                    assert {v['rank'] for v in item['ranks']} == {0, 1}
                    for rank in item['ranks']:
                        assert rank['hybrid_head_modules'] == 1
                        assert rank['gdn_state_dtypes'] == ['torch.float16']
                        assert rank['empty_output_layers'] == rank['fused_attention_layers'] == rank['recurrent_tile8_layers'] == 0
                        assert rank['fused_gdn_quant_layers'] == 48 * accepted
                        assert rank['fused_swiglu_layers'] == 64 * accepted
                        assert any(b['logical_rows'] == b['padded_rows'] == m and not b['has_prefill'] for b in rank['batches'])
                    rates.append(item['output_tokens_per_s'])
                    itls.append(statistics.mean(1000*(v['last_token_ts']-v['first_token_ts'])/1024 for v in item['metrics']))
                    hashes.append(item['token_ids_sha256'])
                    paths.append(sha(path))
                token_hashes.setdefault(m, hashes[0])
                rows.append(dict(rows=m, rates=rates, mean=statistics.mean(rates), sd=statistics.stdev(rates),
                    mean_itl_ms=statistics.mean(itls), token_hashes=hashes,
                    identical_to_p0=all(h == token_hashes[m] for h in hashes), raw_sha256=paths))
            group['decode'].append(dict(block=block, rows=rows, contract=contract, launch=launch))

            label = f'serving-r{block}-{variant}'
            directory = root / label / 'full_gdn'
            launch = read(directory / 'launch.json')
            completed = read(root / f'{label}.complete.json')
            assert completed['library_sha256'] == expected_library[variant]
            for key, value in contract['environment'].items():
                assert launch['environment'][key] == value, key
            for key in ('VLLM_MACH_GDN_EMPTY_OUTPUT', 'VLLM_MACH_FUSED_ATTN_QUANT', 'VLLM_MACH_GDN_STRIDED_BA'):
                assert launch['environment'][key] == '0'
            points = []
            for m in ROWS:
                path = directory / f'c{m}.json'
                item = read(path)
                c = item['contract']
                assert c['input_tokens'] == 3000 and c['output_tokens'] == 1000
                assert c['num_prompts'] == 5*m and c['max_concurrency'] == m
                serving_contracts.setdefault(m, c)
                assert c == serving_contracts[m]
                stats = item['aggregate']
                assert stats['requested'] == stats['completed'] == 5*m
                assert stats['prompt_tokens'] == 15000*m and stats['completion_tokens'] == 5000*m
                assert all(r['success'] for r in item['requests'])
                points.append(dict(concurrency=m, aggregate=stats, contract=c, raw_sha256=sha(path)))
            group['serving'].append(dict(block=block, points=points, launch=launch, completed=completed))

        for m, reference in ((4, 'gdn-m4-fidelity.json'), (32, 'native-fidelity.json')):
            directory = root / f'fidelity-{variant}-m{m}'
            assert read(directory / 'COMPLETE.json')['target_tokens'] == 10479
            contract = read(directory / 'contract.json')
            validate_fidelity_contract(contract, group['decode'][0]['contract'], 'full', 'control', m)
            gdn = read(directory / 'gdn.json')
            assert {r['rank'] for r in gdn} == {0,1}
            assert all(r['prepared_layers'] == 48 and r.get('fused_output_layers',0) == 48*accepted for r in gdn)
            records = read(directory / 'records.json')
            bf16 = read(Path(__file__).with_name(reference))['queries']
            p0 = fidelity_controls[m]
            assert [r['id'] for r in records] == [r['id'] for r in p0] == [q['id'] for q in bf16]
            errors = [float(np.abs(np.asarray(r['gold_logprobs'])-q['bf16_logprobs']).mean()) for r,q in zip(records,bf16,strict=True)]
            differences = [abs(x-y) for r,q in zip(records,p0,strict=True)
                           for x,y in zip(r['gold_logprobs'],q['gold_logprobs'],strict=True)]
            assert len(differences) == 10479
            group['fidelity'][str(m)] = dict(mae=float(np.mean(errors)), ci95=bootstrap(errors),
                max_difference_from_p0=max(differences), repeat=read(directory / 'repeat.json'),
                head=read(directory / 'head.json'), records_sha256=sha(directory / 'records.json'), contract=contract, gdn=gdn)
        data['variants'][variant] = group
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('tp2-full-cumulative.json'))
    a = p.parse_args()
    a.output.write_text(json.dumps(collect(a.root), indent=2) + '\n')
