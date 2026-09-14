from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_mach.exl3 import owner_prefill as owner
from vllm_mach.exl3 import mxfp6_hybrid as hybrid


@pytest.mark.parametrize('metadata,expected', [
    (None, False), ({}, False), ({'attention_only': object()}, False),
    ({'gdn': SimpleNamespace(num_prefills=0)}, False),
    ({'gdn': SimpleNamespace(num_prefills=1)}, True),
])
def test_real_prefill_excludes_v2_autotune_dummy(monkeypatch, metadata, expected):
    monkeypatch.setitem(sys.modules, 'vllm.forward_context', SimpleNamespace(
        get_forward_context=lambda: SimpleNamespace(attn_metadata=metadata)))
    model = SimpleNamespace(layers=[SimpleNamespace(
        layer_type='linear_attention', linear_attn=SimpleNamespace(prefix='gdn'))])
    assert owner.real_prefill(model) is expected


def test_default_replica_layers(monkeypatch):
    monkeypatch.delenv('VLLM_MACH_OWNER_MLP_LAYERS', raising=False)
    assert owner.replica_layers() == tuple(range(0, 64, 2))
    assert len(owner.replica_layers()) * 104448000 == 3342336000


@pytest.mark.parametrize('raw', ['null', '{}', '[0,0]', '[true]', '[-1]', '[64]', '[[1]]'])
def test_invalid_replica_layers(monkeypatch, raw):
    monkeypatch.setenv('VLLM_MACH_OWNER_MLP_LAYERS', raw)
    with pytest.raises(ValueError):
        owner.replica_layers()


def test_disable_replicas(monkeypatch):
    monkeypatch.setenv('VLLM_MACH_OWNER_MLP_LAYERS', '[]')
    assert owner.replica_layers() == ()


def test_every_supported_partition():
    for rows in range(3000, 4097):
        for rank in (0, 1):
            p, counts, own, other = owner.row_partition(rows, rank)
            assert p % 128 == 0 and sum(counts) == rows
            assert 0 < counts[rank] <= p
            assert own.stop - own.start == counts[rank]
            assert {own.start, other.start} == {0, p}
            assert max(own.stop, other.stop) == rows
            assert own.stop == other.start or other.stop == own.start


@pytest.mark.parametrize('rows', [1, 16, 24, 32, 48, 2999, 4097])
def test_non_prefill_fallback(monkeypatch, rows):
    monkeypatch.setattr(owner, 'ENABLED', True)
    assert owner.begin(SimpleNamespace(), torch.empty((rows, owner.H))) is None


@pytest.mark.parametrize('rows', [3000, 3001, 3276, 4095, 4096])
@pytest.mark.parametrize('rank', [0, 1])
def test_padded_final_gather(monkeypatch, rows, rank):
    p, counts, own, _ = owner.row_partition(rows, rank)
    full = torch.arange(rows).reshape(-1, 1)
    packets = torch.zeros((2 * p, 1), dtype=full.dtype)
    packets[:rows] = full
    def gather(packet):
        assert packet.shape == (p, 1)
        assert torch.equal(packet[:counts[rank]], full[own])
        assert not packet[counts[rank]:].count_nonzero()
        return packets
    monkeypatch.setattr(owner, 'gathered_rows', gather)
    state = SimpleNamespace(m=rows, p=p, local_rows=counts[rank])
    assert torch.equal(owner.padded_rows(state, full[own]), full)


@pytest.mark.parametrize('scale_atom', [False, True])
@pytest.mark.parametrize('rows', [3000, 3001, 4095])
@pytest.mark.parametrize('rank', [0, 1])
def test_ragged_transport_crop(monkeypatch, scale_atom, rows, rank):
    p, counts, _, _ = owner.row_partition(rows, rank)
    local = counts[rank]
    awidth, bwidth = (owner.H, owner.H // 32) if scale_atom else (96, 96)
    a = torch.empty(local * awidth, dtype=torch.uint8)
    b = torch.empty(((local + 127) // 128 * 128 if scale_atom else local) * bwidth,
                    dtype=torch.uint8)
    def gather(aa, bb, workspace, oa, ob, actual_rank, workspace_bytes, pdl):
        assert aa is a and bb is b and actual_rank == rank and pdl
        assert oa.numel() == 2 * p * awidth
        assert ob.numel() == 2 * p * bwidth
    monkeypatch.setattr(torch.ops.mach_owner_ragged, 'gather_mx8_padded', gather, raising=False)
    state = SimpleNamespace(m=rows, p=p, local_rows=local, rank=rank,
                            workspace=SimpleNamespace(workspace_tensor=None,
                                                      metadata={'buffer_size': 84213760}))
    av, bv = owner.gather_bytes(state, a, b, scale_atom=scale_atom)
    assert av.numel() == rows * awidth
    assert bv.numel() == ((rows + 127) // 128 * 128 if scale_atom else rows) * bwidth


def test_hybrid_weight_contract():
    weight = hybrid.HybridPackedWeight(torch.empty(1), torch.empty(1), 17408, 5120)
    module = SimpleNamespace(_mach_exl3_mxfp6=hybrid.HybridState(
        route=hybrid.HybridRoute.ALL_ROWS, weights={}, merged_weight=weight))
    assert owner.packed_weight(module) is weight
    module._mach_exl3_mxfp6 = hybrid.HybridState(
        route=hybrid.HybridRoute.ALL_ROWS, weights={None: weight})
    assert owner.packed_weight(module, merged=False) is weight
    with pytest.raises(RuntimeError):
        owner.packed_weight(module)
    with pytest.raises(RuntimeError):
        owner.packed_weight(SimpleNamespace())


def test_native_source_distribution_is_complete():
    root = Path(__file__).resolve().parents[1] / 'native' / 'owner_prefill'
    for source in (*root.glob('*.cu'), *root.glob('*.cuh')):
        for include in re.findall(r'^#include "([^"]+)"', source.read_text(), re.M):
            assert include.startswith('flashinfer/') or (root / include).is_file(), include
    assert '*.cuh' in (root / 'MANIFEST.in').read_text()
    assert 'owner-prefill-v1' in (root / 'abi.cpp').read_text()
    assert not list(root.glob('*.so'))
