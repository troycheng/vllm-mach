"""CPU interface tests only; no model conversion, fitting or GPU execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quantization"))
from common import assets, json_sha


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_rank_pipeline_order_and_paths():
    tool = load("generation_cli", "tools/generate_assets.py")
    args = tool.arguments(["rank64", "--model", "/bf16", "--exl3", "/exl3", "--mxfp6", "/w6",
                           "--calibration-archive", "/longbench.zip", "--output", "/rank64", "--work-dir", "/work"])
    commands = tool.rank_commands(args)
    assert [Path(c[1]).name for c in commands] == ["capture.py", "capture.py", "build_rank64.py"]
    assert "/work/manifests/qa.json" in commands[0]
    assert "/work/manifests/code.json" in commands[1]
    assert "/rank64" in commands[2]


def test_mxfp6_retains_recorded_exclusions():
    tool = load("generation_cli_exclusions", "tools/generate_assets.py")
    assert tool.IGNORE == ["model.language_model.embed_tokens", "lm_head",
                           "re:.*linear_attn.in_proj_a$", "re:.*linear_attn.in_proj_b$",
                           "re:.*visual.*", "re:^mtp.*"]


def test_capture_scheduling_contract():
    capture = load("capture_arguments", "tools/quantization/capture.py")
    args = SimpleNamespace(model=Path("/exl3"), tokenizer=Path("/bf16"))
    short = capture.llm_arguments(args, {"prompt_tokens": 192})
    long = capture.llm_arguments(args, {"prompt_tokens": 3000})
    assert short["max_num_batched_tokens"] == 32 * 191
    assert long["max_num_batched_tokens"] == 4096
    assert long["long_prefill_token_threshold"] == 128
    assert short["max_num_seqs"] == long["max_num_seqs"] == 32
    assert short["enforce_eager"] and long["enforce_eager"]
    assert short["kv_cache_memory_bytes"] == long["kv_cache_memory_bytes"] == 8218214400


def test_frozen_calibration_split_and_offsets():
    data = json.loads((ROOT / "tools/quantization/calibration-selection.json").read_text())
    for kind in ("qa", "code"):
        samples = data[kind]["samples"]
        assert len(samples) == len({x["id"] for x in samples}) == 32
        assert sum(s["split"] == "train" for s in samples) == 16
    qa = data["qa"]["samples"]
    chosen = []
    for task in sorted({s["task"] for s in qa}):
        group = sorted((s["id"] for s in qa if s["task"] == task),
                       key=lambda x: hashlib.sha256(("hessian-pilot-v1:" + x).encode()).digest())
        chosen.extend(group[:2])
    assert set(chosen) == {s["id"] for s in qa if s["split"] == "train"}
    assert data["qa"]["decode_offsets"] == list(range(1, 9))
    assert data["code"]["decode_offsets"] == [1, 128, 256, 384, 512, 640, 768, 1000]


def test_packer_reference_sources_unchanged():
    path = ROOT / "tools/quantization/offline_quant.py"
    tree = ast.parse(path.read_text())
    expected = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "UPSTREAM_SHA256" for t in n.targets))
    assert {name: assets.sha256(path.parent / "upstream" / name) for name in expected} == expected


def test_calibration_rows_use_training_only():
    torch = pytest.importorskip("torch")
    from common import train_rows
    samples = [{"id": str(i), "split": "train" if i < 16 else "holdout", "task": "code"} for i in range(32)]
    offsets = [1,128,256,384,512,640,768,1000]
    x = torch.zeros(32,5120,dtype=torch.bfloat16)
    x[16:] = 9
    data = {"layers": {"language_model.model.layers.0.mlp.gate_up_proj": {"calls": [
        {"decode_offset": offset, "sample_ids": [s["id"] for s in samples], "x": x} for offset in offsets]}}}
    result = train_rows(data, samples, 0, offsets)
    assert result.shape == (128,5120) and result.count_nonzero() == 0
    data["layers"]["language_model.model.layers.0.mlp.gate_up_proj"]["calls"][0]["sample_ids"][0] = "1"
    with pytest.raises(ValueError, match="mapping"):
        train_rows(data,samples,0,offsets)


def test_raw_token_window_digest_is_canonical():
    assert json_sha([1, 2, 3]) == hashlib.sha256(b"[1,2,3]").hexdigest()
