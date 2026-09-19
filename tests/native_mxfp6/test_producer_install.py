"""An unchanged distribution version must not hide an outdated native binary."""
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_mach.mxfp6.install import check_mxfp6_producers


@pytest.mark.parametrize("missing", [None, "python_gdn", "native_swiglu", "native_pdl"])
def test_required_producers_checked_before_install_or_launch(monkeypatch, missing):
    loaded = []
    extension = SimpleNamespace(
        load_library=lambda: loaded.append(True),
        gemm_from_swiglu=lambda: None,
        gemm_from_gdn=lambda: None,
    )
    operations = SimpleNamespace(
        silu_and_mul_mxfp8=object(), gemm_from_swiglu=object(), gemm_w6a8_pdl=object()
    )
    if missing == "python_gdn":
        del extension.gemm_from_gdn
    elif missing == "native_swiglu":
        del operations.gemm_from_swiglu
    elif missing == "native_pdl":
        del operations.gemm_w6a8_pdl
    monkeypatch.setitem(sys.modules, "mxfp6", extension)
    monkeypatch.setattr(torch.ops, "mxfp6", operations)
    if missing:
        with pytest.raises(RuntimeError, match="force-reinstall"):
            check_mxfp6_producers()
    else:
        check_mxfp6_producers()
    assert loaded == [True]
