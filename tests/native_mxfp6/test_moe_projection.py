"""CPU coverage for the Qwen3.5 projection override admission boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from vllm_mach.mxfp6 import moe_projection as projection


class _Ops:
    def __init__(self, abi="native-w6a8-30-v5", accepted=True):
        self.abi = abi
        self.accepted = accepted
        self.calls = []

    def w6a8_config_abi(self, anchor):
        return self.abi

    def set_w6a8_config(self, anchor, m, n, k, config_id, swizzle, raster, dtype):
        self.calls.append((m, n, k, config_id, swizzle, raster, dtype))
        return self.accepted


@pytest.fixture
def eligible(monkeypatch):
    monkeypatch.delenv("VLLM_MACH_MOE_PROJECTION_TUNING", raising=False)
    config = NS(**projection._GEOMETRY, model_type="qwen3_5_moe_text")
    model = NS(config=NS(text_config=config))
    parallel = NS(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        use_ubatching=False,
    )
    vllm_config = NS(parallel_config=parallel, speculative_config=None, lora_config=None)
    model.vllm_config = vllm_config
    device = NS(type="cuda", index=1)
    problems = [(n, k, NS(device=device), object()) for n, k in projection._SHAPES]
    ops = _Ops()
    extension = NS(load_library=lambda: None)
    monkeypatch.setattr(projection, "_import_mxfp6", lambda: extension)
    monkeypatch.setattr(projection.torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(projection.torch, "empty", lambda *_, **__: NS())
    monkeypatch.setattr(projection.torch.ops, "mxfp6", ops, raising=False)
    autotune = ModuleType("mxfp6.autotune")
    autotune.is_autotune_enabled = lambda: False
    monkeypatch.setitem(sys.modules, "mxfp6.autotune", autotune)
    return NS(model=model, config=config, vllm_config=vllm_config, problems=problems, ops=ops)


def test_packaged_table_is_the_frozen_37_entry_projection_set():
    table = json.loads(Path(projection._TABLE).read_text())
    entries = table["entries"]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    assert table["native_config_abi"] == "native-w6a8-30-v5"
    assert len(entries) == 37
    assert hashlib.sha256(canonical).hexdigest() == (
        "c1c5b47546c6d4d0c8f77a7d1d6ae3a96d3eeb7fb0c0b4ca96e2ca6165d9c110"
    )
    tuples = [(row["m"], row["n"], row["k"]) for row in entries]
    assert len(tuples) == len(set(tuples))
    assert {(row["n"], row["k"]) for row in entries} == projection._SHAPES


def test_eligible_rank_registers_each_frozen_row_once(eligible):
    assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 37
    table = json.loads(Path(projection._TABLE).read_text())
    assert eligible.ops.calls == [
        (row["m"], row["n"], row["k"], *row["config"], 0, torch.bfloat16)
        for row in table["entries"]
    ]


def test_abi_is_checked_before_any_registration(eligible):
    eligible.ops.abi = "native-w6a8-30-v4"
    with pytest.raises(RuntimeError, match="config ABI mismatch"):
        projection.prepare(eligible.model, eligible.problems, torch.bfloat16)
    assert not eligible.ops.calls


@pytest.mark.parametrize("field,value", [("model_type", "qwen3_5"), ("hidden_size", 4096)])
def test_dense_or_wrong_model_config_is_excluded(eligible, field, value):
    setattr(eligible.config, field, value)
    assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 0
    assert not eligible.ops.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 1),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_expert_parallel", True),
        ("enable_eplb", True),
        ("use_sequence_parallel_moe", True),
        ("use_ubatching", True),
    ],
)
def test_parallel_incompatible_models_are_excluded(eligible, field, value):
    setattr(eligible.vllm_config.parallel_config, field, value)
    assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 0
    assert not eligible.ops.calls


@pytest.mark.parametrize("field", ["speculative_config", "lora_config"])
def test_request_features_are_excluded(eligible, field):
    setattr(eligible.vllm_config, field, object())
    assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 0
    assert not eligible.ops.calls


@pytest.mark.parametrize("kind,capability,dtype", [("cpu", (12, 0), torch.bfloat16), ("cuda", (9, 0), torch.bfloat16), ("cuda", (12, 0), torch.float16)])
def test_dtype_and_device_admission(eligible, monkeypatch, kind, capability, dtype):
    n, k, *_ = eligible.problems[0]
    eligible.problems[0] = (n, k, NS(device=NS(type=kind, index=1)), object())
    monkeypatch.setattr(projection.torch.cuda, "get_device_capability", lambda _: capability)
    assert projection.prepare(eligible.model, eligible.problems, dtype) == 0
    assert not eligible.ops.calls


def test_invalid_mode_and_disabled_mode(eligible, monkeypatch):
    monkeypatch.setenv("VLLM_MACH_MOE_PROJECTION_TUNING", "bad")
    with pytest.raises(ValueError, match="auto, 0 or 1"):
        projection.prepare(eligible.model, eligible.problems, torch.bfloat16)
    monkeypatch.setenv("VLLM_MACH_MOE_PROJECTION_TUNING", "0")
    assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 0
    assert not eligible.ops.calls


@pytest.mark.parametrize("mode,raises", [("auto", False), ("1", True)])
def test_autotune_auto_skips_and_explicit_mode_rejects(eligible, monkeypatch, mode, raises):
    autotune = ModuleType("mxfp6.autotune")
    autotune.is_autotune_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "mxfp6.autotune", autotune)
    monkeypatch.setenv("VLLM_MACH_MOE_PROJECTION_TUNING", mode)
    if raises:
        with pytest.raises(RuntimeError, match="MXFP6_AUTOTUNE=off"):
            projection.prepare(eligible.model, eligible.problems, torch.bfloat16)
    else:
        assert projection.prepare(eligible.model, eligible.problems, torch.bfloat16) == 0
    assert not eligible.ops.calls


def test_native_rejection_is_not_silently_accepted(eligible):
    eligible.ops.accepted = False
    with pytest.raises(RuntimeError, match="Could not register"):
        projection.prepare(eligible.model, eligible.problems, torch.bfloat16)


def test_missing_native_api_is_rejected(eligible, monkeypatch):
    monkeypatch.setattr(projection.torch.ops, "mxfp6", NS())
    with pytest.raises(RuntimeError, match="updated mxfp6-sm120"):
        projection.prepare(eligible.model, eligible.problems, torch.bfloat16)


def test_conditional_and_language_only_models_share_admission(eligible):
    language = NS(config=eligible.config, vllm_config=eligible.vllm_config)
    assert projection.prepare(language, eligible.problems, torch.bfloat16) == 37
    assert projection.prepare(NS(language_model=language), eligible.problems, torch.bfloat16) == 37
