"""Reject mislabeled full measurements before comparing precision profiles."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm_mach.mxfp6.serve import profile_environment

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('combination_collector',
    ROOT / 'docs/data/collect_tp2_combinations.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def contract(profile):
    archived = json.loads((ROOT / 'docs/data/tp2-p2b.json').read_text())['matched_control']['decode']['contract']
    result = copy.deepcopy(archived)
    full = profile == 'full'
    result['precision_profile'] = profile
    result['config']['mamba_ssm_cache_dtype'] = 'float16' if full else 'float32'
    result['environment'] = profile_environment(SimpleNamespace(fp16_ssm=full,
        lossless_prefill=full, owner_prefill=full, nvfp4_lm_head=full, verify_prefill=False))
    return result


@pytest.mark.parametrize('profile', ['default', 'full'])
def test_precision_control_contract(profile):
    collector.validate_contract(contract(profile), profile, False, False)


@pytest.mark.parametrize('flag', ['VLLM_QWEN3_5_FP16_SSM', 'VLLM_SM120_LOSSLESS_PREFILL',
                                'VLLM_SM120_OWNER_PREFILL', 'VLLM_HYBRID_NVFP4_LM_HEAD'])
def test_full_requires_every_full_option(flag):
    data = contract('full')
    data['environment'][flag] = '0'
    with pytest.raises(AssertionError):
        collector.validate_contract(data, 'full', False, False)


def test_full_rejects_fp32_cache_despite_flag():
    data = contract('full')
    data['config']['mamba_ssm_cache_dtype'] = 'float32'
    with pytest.raises(AssertionError):
        collector.validate_contract(data, 'full', False, False)


def test_combination_rejects_control_results():
    with pytest.raises(AssertionError):
        collector.validate_contract(contract('full'), 'full', True, True)


@pytest.mark.parametrize('change', ['arm', 'mamba_ssm_cache_dtype', 'extension_library_sha256'])
def test_fidelity_rejects_unmatched_full_results(change):
    baseline = contract('full')
    fidelity = dict(baseline, arm='full_gdn', physical_rows=4,
                    llm_args=dict(baseline['config'], logprobs_mode='raw_logprobs'))
    collector.validate_fidelity_contract(fidelity, baseline, 'full', 'control', 4)
    if change == 'mamba_ssm_cache_dtype':
        fidelity['llm_args'][change] = 'float32'
    else:
        fidelity[change] = 'different'
    with pytest.raises(AssertionError):
        collector.validate_fidelity_contract(fidelity, baseline, 'full', 'control', 4)
