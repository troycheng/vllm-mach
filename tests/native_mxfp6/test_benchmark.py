"""Matched request content and arrival schedule are independent of model arm."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location(
        "serving_benchmark", ROOT / "tools/benchmark_native_mxfp6.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    return argparse.Namespace(
        **(
            dict(
                contract_seed=20260915,
                token_id_low=1000,
                token_id_high=240000,
                input_tokens=3000,
                output_tokens=1000,
                num_prompts=20,
                max_concurrency=4,
                request_seed_base=2026091500,
                request_rate=100,
                top_k=20,
                top_p=0.95,
                prompt_manifest=ROOT / "docs/data/serving-prompts.json",
            )
            | overrides
        )
    )


def test_frozen_serving_prompts_and_arrivals():
    tool = load()
    first, prompts = tool.make_contract(args())
    second, again = tool.make_contract(args())
    assert first == second and prompts == again
    assert len(prompts) == 20 and all(len(p) == 3000 for p in prompts)
    assert first["arrival_offsets_s"] == sorted(first["arrival_offsets_s"])
    assert first["arrival_offsets_s"][-1] > 0
    assert first["sampling"]["top_k"] == 20 and first["sampling"]["top_p"] == 0.95


def test_uniform_legacy_workload_is_available():
    contract, prompts = load().make_contract(args(prompt_manifest=None, request_rate=0))
    assert contract["arrival_offsets_s"] == [0.0] * 20
    assert contract["prompt_source"] == "uniform token IDs"
    assert all(1000 <= token < 240000 for p in prompts for token in p)


def test_manifest_must_cover_all_requests(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"prompts": [{"token_ids": [1, 2]}]}))
    with pytest.raises(AssertionError):
        load().make_contract(args(prompt_manifest=path))


@pytest.mark.parametrize("arm", ["fp8", "nvfp4"])
def test_stock_baseline_keeps_compilation_and_disables_flashinfer_ar(monkeypatch, arm):
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "1")
    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "1")
    command, env = tool.launch_configuration(
        argparse.Namespace(
            models=Path("/models"),
            stock_runtime=Path("/stock"),
            devices="0,1",
            port=8000,
        ),
        arm,
    )
    assert "--compilation-config" not in command
    assert "--kv-cache-memory-bytes" not in command
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert env["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0" and env["VLLM_PLUGINS"] == ""
    assert "VLLM_QWEN3_5_FUSED_AR_NORM" not in env
    assert env["PYTHONPATH"] == "/stock"


def test_published_serving_counts_and_throughput():
    data = json.loads((ROOT / "docs/data/native-serving.json").read_text())
    assert set(data["runs"]) == {"fp8", "nvfp4", "default", "full"}
    columns = data["request_columns"]
    for arm, run in data["runs"].items():
        assert [p["concurrency"] for p in run["points"]] == [4, 16, 24, 32]
        for point in run["points"]:
            c = point["concurrency"]
            contract = data["contracts"][str(c)]
            assert point["warmup"] == {"requests": min(32, 5 * c), "output_tokens": 128}
            rows = [
                dict(zip(columns, row, strict=True)) for row in point["request_rows"]
            ]
            assert len(rows) == point["completed"] == point["requested"] == 5 * c
            assert {r["request_index"] for r in rows} == set(range(5 * c))
            assert all(r["success"] and r["http_status"] == 200 for r in rows)
            assert all(
                r["prompt_tokens"] == 3000 and r["completion_tokens"] == 1000
                for r in rows
            )
            assert point["completion_tokens"] == sum(
                r["completion_tokens"] for r in rows
            )
            assert point["prompt_tokens"] == sum(r["prompt_tokens"] for r in rows)
            assert point["output_throughput_tokens_per_s"] == pytest.approx(
                point["completion_tokens"] / point["duration_s"]
            )
            assert contract["max_concurrency"] == c and contract["num_prompts"] == 5 * c
        if arm in ("fp8", "nvfp4"):
            assert run["launch"]["environment"]["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
            assert run["launch"]["environment"]["VLLM_PLUGINS"] == ""
            assert "--compilation-config" not in run["launch"]["command"]
