"""Validate exact attention gate fusion, both-rank routing and matched trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from summarize_tp2_decode import summarize_trace

ROWS = (1, 2, 4, 8, 16, 24, 32)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trial(path, m, expected_layers):
    result = read(path)
    tokens = 129 if m == 1 else 1025
    assert len(result['metrics']) == m
    assert all(v['num_generation_tokens'] == tokens and not v['is_corrupted']
               for v in result['metrics'])
    assert {r['rank'] for r in result['ranks']} == {0, 1}
    for rank in result['ranks']:
        assert rank['fused_attention_layers'] == expected_layers
        assert rank['fused_swiglu_layers'] == 64
        assert rank['fused_gdn_quant_layers'] == 48
        assert any(b['logical_rows'] == b['padded_rows'] == m and not b['has_prefill']
                   for b in rank['batches'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('tp2-attention-validation.json'))
    a = p.parse_args()
    trials, traces = [], []
    contracts = {mode: read(a.root / mode / 'contract.json') for mode in ('off', 'on')}
    for key in contracts['off']:
        if key != 'fused_attention_quant':
            assert contracts['off'][key] == contracts['on'][key], key
    for mode, expected_layers in [('off', 0), ('on', 16)]:
        assert contracts[mode]['fused_attention_quant'] == str(int(mode == 'on'))
        assert contracts[mode]['repeats'] == 5 and not contracts[mode]['profile']
        for m in ROWS:
            for trial in range(5):
                path = a.root / mode / f'm{m}-trial{trial}/result.json'
                result = validate_trial(path, m, expected_layers)
                trials.append(dict(mode=mode, rows=m, trial=trial, sha256=sha(path),
                                   output_tokens_per_s=result['output_tokens_per_s']))
            directory = a.root / f'{mode}-profile/m{m}-trial0'
            ranks = read(directory / 'result.json')['ranks']
            assert {r['rank'] for r in ranks} == {0, 1}
            for rank in ranks:
                assert rank['fused_attention_layers'] == expected_layers
                assert len(rank['samples']) == 3
                assert all(s['logical_rows'] == s['padded_rows'] == m for s in rank['samples'])
                path = directory / f'rank{rank["rank"]}.json'
                categories = summarize_trace(path)['categories']
                kernels = [e for e in read(path)['traceEvents'] if e.get('cat') == 'kernel']
                sigmoid = [e for e in kernels if 'sigmoid_kernel_cuda' in e['name']
                           and 'BFloat16' in e['name']]
                multiply = [e for e in kernels if 'BinaryFunctor<c10::BFloat16' in e['name']
                            and 'MulFunctor' in e['name']]
                assert len(sigmoid) == len(multiply) == 3 * (16 - expected_layers)
                assert categories.get('attention_gate_quantization', {}).get('count', 0) == 3 * expected_layers
                assert categories['activation_quantization']['count'] == 3 * (144 - expected_layers)
                assert categories['allreduce_residual_norm']['count'] == 384
                traces.append(dict(mode=mode, rows=m, rank=rank['rank'], categories=categories,
                                   sha256=sha(path), sigmoid_launches=len(sigmoid),
                                   multiply_launches=len(multiply),
                                   sigmoid_summed_us=sum(e['dur'] for e in sigmoid),
                                   multiply_summed_us=sum(e['dur'] for e in multiply)))
    reverse = []
    reverse_contracts = {mode: read(a.root / f'reverse-{mode}/contract.json') for mode in ('off', 'on')}
    for mode, expected_layers in [('off', 0), ('on', 16)]:
        for key in contracts[mode]:
            if key != 'devices':
                assert reverse_contracts[mode][key] == contracts[mode][key], key
        assert reverse_contracts[mode]['devices'] == '6,7'
        summary = read(a.root / f'reverse-{mode}/summary.json')
        assert len(summary) == 1 and summary[0]['rows'] == 32
        trial_hashes = []
        for trial in range(5):
            path = a.root / f'reverse-{mode}/m32-trial{trial}/result.json'
            validate_trial(path, 32, expected_layers)
            trial_hashes.append(sha(path))
        reverse.append(dict(mode=mode, **{k:v for k,v in summary[0].items() if k != 'trials'},
                            trial_sha256=trial_hashes, contract=reverse_contracts[mode]))
    fidelity = {}
    previous = a.root.parents[1] / 'tp2-optimization-20260916'
    for m in (4, 32):
        current = a.root / f'fidelity-m{m}'
        assert (a.root / f'fidelity-m{m}.log').read_text().count(
            'Mach prepared 16 TP2 attention/MXFP8 producers') == 2
        assert read(current / 'COMPLETE.json')['target_tokens'] == 10479
        contract = read(current / 'contract.json')
        assert contract['fused_attention_quant'] == '1'
        records = read(current / 'records.json')
        accepted = read(previous / f'p2a-fused-fidelity-m{m}/records.json')
        assert len(records) == len(accepted) == 256
        assert [r['id'] for r in records] == [r['id'] for r in accepted]
        errors = [abs(x-y) for r, old in zip(records, accepted, strict=True)
                  for x, y in zip(r['gold_logprobs'], old['gold_logprobs'], strict=True)]
        assert len(errors) == 10479
        fidelity[str(m)] = dict(max_difference_from_accepted=max(errors),
            mean_difference_from_accepted=sum(errors)/len(errors),
            records_sha256=sha(current / 'records.json'), repeat=read(current / 'repeat.json'))
    logs = {}
    for name, count in [('gpu-tests.log', 23), ('integration-tests.log', 25)]:
        path = a.root / name
        assert f'{count} passed' in path.read_text()
        logs[name] = dict(passed=count, sha256=sha(path))
    data = dict(schema='mach-tp2-attention-validation/v1', contracts=contracts,
        trials=trials, reverse_order=reverse, traces=traces, fidelity=fidelity, tests=logs,
        packaging=read(a.root / 'packaging.json'), runtime=read(a.root / 'runtime.json'))
    a.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
