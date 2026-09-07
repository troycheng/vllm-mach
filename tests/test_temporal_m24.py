from types import SimpleNamespace

import pytest

from vllm_mach.exl3 import temporal_m24


@pytest.mark.parametrize("rows", [1, 4, 16, 23, 24, 25, 32, 128])
@pytest.mark.parametrize("count,n", [(8, 1024), (14, 512)])
def test_only_exact_m24_bundles(rows, count, n):
    assert temporal_m24.eligible(rows, 5120, 6, count, n) == (rows == 24)
    assert not temporal_m24.eligible(rows, 5120, 5, count, n)
    assert not temporal_m24.eligible(rows, 4096, 6, count, n)
    assert not temporal_m24.eligible(rows, 5120, 6, count, n + 128)


def test_missing_extension_falls_back(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name, name=name)
    temporal_m24.load_extension.cache_clear()
    monkeypatch.setattr(temporal_m24.importlib, "import_module", missing)
    assert temporal_m24.load_extension() is None
    temporal_m24.load_extension.cache_clear()


def test_broken_extension_is_not_silently_ignored(monkeypatch):
    temporal_m24.load_extension.cache_clear()
    monkeypatch.setattr(temporal_m24.importlib, "import_module", lambda _: SimpleNamespace())
    with pytest.raises(RuntimeError, match="lacks run_grouped"):
        temporal_m24.load_extension()
    temporal_m24.load_extension.cache_clear()
