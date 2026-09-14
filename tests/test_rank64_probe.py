"""The diagnostic sampler must reject reused graph buffers before recording."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/rank64_probe.py'
    spec = importlib.util.spec_from_file_location('rank64_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return module


def test_snapshot_is_atomic_on_nonfinite_input(probe, monkeypatch):
    states = [SimpleNamespace(graph_input=torch.ones(32, 8)),
              SimpleNamespace(graph_input=torch.full((32, 8), float('nan')))]
    monkeypatch.setattr(probe, '_states', lambda _: states)
    with pytest.raises(ValueError, match='steady M32'):
        probe.snapshot_worker_inputs(None)
    assert all(not hasattr(s, '_probe_inputs') for s in states)


def test_snapshot_retains_only_two_independent_copies(probe, monkeypatch):
    state = SimpleNamespace(graph_input=torch.ones(32, 8))
    monkeypatch.setattr(probe, '_states', lambda _: [state])
    for value in (1, 2, 3):
        state.graph_input.fill_(value)
        probe.snapshot_worker_inputs(None)
    assert len(state._probe_inputs) == 2
    assert torch.all(state._probe_inputs[0] == 2)
    assert torch.all(state._probe_inputs[1] == 3)
