"""Mach's graph-changing settings must participate in vLLM's AOT lookup."""
import pytest


@pytest.mark.parametrize('name,value', [
    ('VLLM_MACH_FUSED_GEMMA_NORM', '0'),
    ('VLLM_MACH_MXFP8_BACKEND', 'flashinfer'),
    ('VLLM_MACH_MXFP8_FUSED_MLP', '0'),
    ('VLLM_MACH_MXFP8_NORM_QUANT', '1'),
    ('VLLM_MACH_MXFP8_PDL', '1'),
    ('VLLM_MACH_GDN_TP1', '1'),
    ('VLLM_MACH_GDN_PERSISTENT', '1'),
    ('VLLM_MACH_GDN_BA_OVERLAP', '1'),
])
def test_compile_factors_include_adapter_switches(monkeypatch, name, value):
    from vllm import envs

    from vllm_mach.plugin import register_compile_factors

    monkeypatch.setattr(envs, 'environment_variables', dict(envs.environment_variables))
    monkeypatch.delenv(name, raising=False)
    register_compile_factors()
    before = envs.compile_factors()
    monkeypatch.setenv(name, value)
    after = envs.compile_factors()
    assert before[name] != after[name]
    assert after[name] == value
    assert len(after['VLLM_MACH_CODE_HASH']) == 64
    assert before['VLLM_MACH_CODE_HASH'] == after['VLLM_MACH_CODE_HASH']
