from types import SimpleNamespace

import pytest
import torch

from vllm_mach.exl3 import dense_adapter as adapter
from vllm_mach.exl3 import m32


def bundle(grouped=True):
    return (torch.empty(8), None, None, 6, True, False, 128,
            torch.empty(1 if grouped else 8))


@pytest.mark.parametrize("rows", [0, 1, 4, 16, 17, 24, 31, 32, 33, 128])
@pytest.mark.parametrize("grouped", [False, True])
def test_bundle_row_dispatch(monkeypatch, rows, grouped):
    monkeypatch.setattr(adapter, "_BF16_IO_ENABLED", True)
    monkeypatch.setattr(adapter, "_BF16_IO_M24_ENABLED", True)
    monkeypatch.setattr(adapter, "_BF16_IO_M32_ENABLED", True)
    assert adapter._bf16_bundle_rows_supported(rows, bundle(grouped)) == (
        1 <= rows <= 16 or (grouped and rows in (24, 32))
    )
    monkeypatch.setattr(adapter, "_BF16_IO_ENABLED", False)
    assert not adapter._bf16_bundle_rows_supported(rows, bundle(grouped))


@pytest.mark.parametrize("rows,expected_chunks", [(24, [16, 8]), (32, [16, 16])])
def test_split_rows_write_one_output(monkeypatch, rows, expected_chunks):
    calls = []

    def grouped(x, ptrs, scratch, out, unique, had, svh, ids, *args):
        calls.append((x.shape[0], out.untyped_storage().data_ptr()))
        assert scratch.shape == (8, x.shape[0], 128)
        assert had.shape == (1, x.shape[0], 128)
        assert x.is_contiguous() and out.is_contiguous()
        out.copy_(x.repeat(1, 8))

    monkeypatch.setattr(adapter, "_load_exl3_ext",
                        lambda: SimpleNamespace(exl3_mgemm_bf16_io_grouped_had=grouped))
    monkeypatch.setattr(adapter, "_BF16_IO_TILE_M32_ENABLED", False)
    x = torch.arange(rows, dtype=torch.bfloat16).view(rows, 1).expand(rows, 128).contiguous()
    ptrs = torch.empty(8, dtype=torch.int64)
    out = adapter._exl3_mgemm_bf16_io._init_fn(
        x, ptrs, ptrs, ptrs, torch.empty(1, dtype=torch.int64),
        torch.zeros(8, dtype=torch.int32), 6, 128)
    assert [c[0] for c in calls] == expected_chunks
    assert len({c[1] for c in calls}) == 1
    assert torch.equal(out, x.repeat(1, 8))


def test_optional_extension_absence_and_abi(monkeypatch):
    m32.load_extension.cache_clear()
    def missing(name):
        raise ModuleNotFoundError(name=name)
    monkeypatch.setattr(m32.importlib, "import_module", missing)
    assert m32.load_extension() is None
    m32.load_extension.cache_clear()
    monkeypatch.setattr(m32.importlib, "import_module", lambda name: object())
    with pytest.raises(RuntimeError, match="lacks grouped_had_m32"):
        m32.load_extension()
    m32.load_extension.cache_clear()


def test_large_rows_require_individual_opt_in(monkeypatch):
    monkeypatch.setattr(adapter, "_BF16_IO_ENABLED", True)
    monkeypatch.setattr(adapter, "_BF16_IO_M24_ENABLED", False)
    monkeypatch.setattr(adapter, "_BF16_IO_M32_ENABLED", False)
    monkeypatch.setattr(adapter, "_BF16_IO_TILE_M32_ENABLED", True)
    assert not adapter._bf16_bundle_rows_supported(24, bundle())
    assert not adapter._bf16_bundle_rows_supported(32, bundle())
    monkeypatch.setattr(adapter, "_BF16_IO_M32_ENABLED", True)
    assert adapter._bf16_bundle_rows_supported(32, bundle())
    assert not adapter._bf16_bundle_rows_supported(24, bundle())


def test_extension_transitive_import_error_is_not_hidden(monkeypatch):
    m32.load_extension.cache_clear()
    def missing_dependency(name):
        raise ModuleNotFoundError(name="missing_dependency")
    monkeypatch.setattr(m32.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError):
        m32.load_extension()
    m32.load_extension.cache_clear()


def test_true_m32_workspace(monkeypatch):
    calls = []
    def kernel(x, ptrs, scratch, out, unique, had, svh, ids, locks, *args):
        assert locks.dtype == torch.int32 and locks.numel() == 8
        assert had.shape == (1, 32, 128)
        calls.append(locks)
        out.fill_(7)
    monkeypatch.setattr(adapter, "_load_exl3_ext", lambda: object())
    monkeypatch.setattr(adapter, "_load_exl3_m32_ext",
                        lambda: SimpleNamespace(grouped_had_m32=kernel))
    monkeypatch.setattr(adapter, "_BF16_IO_TILE_M32_ENABLED", True)
    ptrs = torch.empty(8, dtype=torch.int64)
    x = torch.zeros(32, 128, dtype=torch.bfloat16)
    outputs = [adapter._exl3_mgemm_bf16_io._init_fn(
        x, ptrs, ptrs, ptrs, torch.empty(1, dtype=torch.int64),
        torch.zeros(8, dtype=torch.int32), 6, 128) for _ in range(2)]
    assert calls[0].data_ptr() != calls[1].data_ptr()
    assert all(torch.all(out == 7) for out in outputs)
