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


def test_standalone_native_keeps_main_kernel_selection(monkeypatch):
    """Generic registration must not shadow the standalone FlashInfer adapter."""
    from types import SimpleNamespace
    from vllm_mach import mxfp6

    imports = []
    def load(name):
        imports.append(name)
        assert name == 'vllm_mach.mxfp6.dense'
        return SimpleNamespace(register_dense_kernel=lambda: True)

    monkeypatch.setenv('VLLM_MACH_NATIVE_MXFP8', '1')
    monkeypatch.setattr(mxfp6, '_optional_runtime_is_installed', lambda: True)
    monkeypatch.setattr(mxfp6, 'importlib', SimpleNamespace(import_module=load))
    assert mxfp6.register_dense_kernel()
    assert imports == ['vllm_mach.mxfp6.dense']


def test_unavailable_runtime_keeps_plugin_import_light(monkeypatch):
    from vllm_mach import plugin

    monkeypatch.delenv('VLLM_MACH_PROFILE', raising=False)
    monkeypatch.delenv('VLLM_MACH_NATIVE_MXFP8', raising=False)
    monkeypatch.setattr(plugin, 'version', lambda name: '0.29.0')
    monkeypatch.setattr(plugin, 'register_dense_kernel', lambda: False)
    def unexpected():
        pytest.fail('optional runtime opt-out must not import compile/CUDA code')
    monkeypatch.setattr(plugin, 'register_compile_factors', unexpected)
    plugin.register()


@pytest.mark.parametrize('official,version', [(True, None), (False, 3)])
def test_pdl_accepts_qualified_native_abis(monkeypatch, official, version):
    from types import SimpleNamespace as NS
    from vllm_mach.mxfp6 import dense_mxfp8 as adapter
    ops = NS(gemm=NS(default=NS(_schema='gemm(bool use_pdl=False)' if official else 'gemm()')),
             gemm_pdl=object(), gemm_from_float_pdl=object())
    if version is not None:
        ops.pdl_version = lambda: version
    native = NS(quantize_mxfp8_pdl=object())
    monkeypatch.setattr(adapter, 'torch', NS(ops=NS(mxfp8_sm120=ops, mxfp6=native)))
    adapter._require_pdl_runtime.cache_clear()
    try:
        adapter._require_pdl_runtime()
        del native.quantize_mxfp8_pdl
        adapter._require_pdl_runtime.cache_clear()
        with pytest.raises(RuntimeError, match='both rebuilt native libraries'):
            adapter._require_pdl_runtime()
    finally:
        adapter._require_pdl_runtime.cache_clear()
