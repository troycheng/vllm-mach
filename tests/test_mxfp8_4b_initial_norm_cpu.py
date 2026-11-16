"""The initial embedding norm uses the qualified GROUP4 order only for 4B BF16."""
import ast
import json
from pathlib import Path
import pytest
from vllm_mach.mxfp8.two_b import ordered_reduction as ordering
from vllm_mach.gdn_prep import four_b_enabled

SOURCE = (Path(__file__).parent / 'fixtures/mxfp8_4b_initial_norm.py').read_text()
NAME = 'triton_red_fused__to_copy_add_embedding_rms_norm_0'


@pytest.mark.parametrize('mode,rows', [('production', None), ('quality', '32'), ('quality', '64')])
def test_exact_component_source_all_modes(monkeypatch, mode, rows):
    monkeypatch.setenv('VLLM_MACH_PROFILE', 'qwen35-4b-mxfp8-champion-v1')
    monkeypatch.setenv('VLLM_MACH_MXFP8_SSM_DTYPE', 'bfloat16')
    monkeypatch.setenv('VLLM_MACH_MXFP8_MODE', mode)
    monkeypatch.setenv('VLLM_MACH_MXFP8_ORDERED_BF16_SSM', '1')
    source = SOURCE.replace("'x': 8192", "'x': " + str(2048 if rows is None else int(rows)*256))
    changed = ordering.rewrite(source, NAME)
    target = next(iter(ordering.policy()['targets'].values()))
    assert ordering.canonical_function_sha256(changed, NAME) == target['after_sha256']
    assert target['after_sha256'] == '4480da6f4c73b10ec6a6ba9e7b2d555ae8bc9a0ce2b04bd9b02566ac4cd855ec'
    assert ordering.rewrite(changed, NAME) == changed
    assert target['selected']['kwargs'] == {'XBLOCK': 1, 'R0_BLOCK': 1024}
    with pytest.raises(RuntimeError):
        ordering.rewrite(source.replace('1e-06', '2e-06'), NAME)


@pytest.mark.parametrize('dtype,ordered,enabled', [('float32','1',False), ('bfloat16','0',False), ('bfloat16','1',True)])
def test_profile_scope(monkeypatch, dtype, ordered, enabled):
    monkeypatch.setenv('VLLM_MACH_PROFILE', 'qwen35-4b-mxfp8-champion-v1')
    monkeypatch.setenv('VLLM_MACH_MXFP8_SSM_DTYPE', dtype)
    monkeypatch.setenv('VLLM_MACH_MXFP8_ORDERED_BF16_SSM', ordered)
    assert four_b_enabled() is enabled
    assert (ordering.policy_path().name == 'initial_norm_bf16.json') is enabled
