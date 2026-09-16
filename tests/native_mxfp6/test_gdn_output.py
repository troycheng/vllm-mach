"""Legacy extension must not select an unavailable fused GDN producer."""
from types import SimpleNamespace
import pytest
from vllm_mach.mxfp6 import gdn_output


@pytest.mark.parametrize('mode', ['auto','0','1'])
def test_legacy_extension(monkeypatch,mode):
    monkeypatch.setenv('VLLM_MACH_FUSED_GDN_QUANT',mode)
    monkeypatch.setattr(gdn_output,'_eligible',lambda layer: True)
    monkeypatch.setattr(gdn_output,'_import_mxfp6',lambda: SimpleNamespace(load_library=lambda: None))
    layer=SimpleNamespace()
    if mode=='1':
        with pytest.raises(RuntimeError,match='gemm_from_gdn'):
            gdn_output.prepare(layer)
    else:
        assert not gdn_output.prepare(layer)
    assert not hasattr(layer,'_mach_gdn_output_fused')


def test_mode_validation(monkeypatch):
    monkeypatch.setenv('VLLM_MACH_FUSED_GDN_QUANT','invalid')
    with pytest.raises(ValueError,match='auto, 0 or 1'):
        gdn_output.prepare(SimpleNamespace())
