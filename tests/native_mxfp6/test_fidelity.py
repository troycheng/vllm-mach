"""CPU regression checks for the numerical experiment and published data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path):
    spec = importlib.util.spec_from_file_location("fidelity_test_module", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_manifest():
    tool = module("tools/fidelity_native_mxfp6.py")
    samples = tool.manifest(ROOT / "docs/data/fidelity-samples.json")
    assert len(samples) == 256
    assert sorted({s["domain"] for s in samples}) == [
        "chinese",
        "code",
        "english",
        "math",
    ]
    assert all(
        sum(s["domain"] == domain for s in samples) == 64
        for domain in {s["domain"] for s in samples}
    )


def test_manifest_rejects_token_mismatch(tmp_path):
    tool = module("tools/fidelity_native_mxfp6.py")
    data = json.loads((ROOT / "docs/data/fidelity-samples.json").read_text())
    data["samples"][0]["target_token_ids"][0] += 1
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        tool.manifest(path)


def test_query_bootstrap_and_paired_delta():
    tool = module("docs/data/collect_native_comparison.py")
    assert tool.bootstrap([0.25] * 256) == [0.25, 0.25]
    assert tool.bootstrap([0.0] * 256) == [0.0, 0.0]


def test_published_fidelity():
    path = ROOT / "docs/data/native-fidelity.json"
    if not path.exists():
        pytest.skip("GPU experiment results not collected yet")
    data = json.loads(path.read_text())
    assert set(data["runs"]) == {
        "fp8",
        "default",
        "persistent",
        "gdn",
        "full",
        "full_ba",
        "full_gdn",
        "nvfp4",
    }
    assert data["query_count"] == len(data["queries"]) == 256
    assert sum(len(q["bf16_logprobs"]) for q in data["queries"]) == 10479
    assert data["reference_repeat"] == {"max_abs": 0.0, "mean_abs": 0.0}
    tool = module("docs/data/collect_native_comparison.py")
    for run in data["runs"].values():
        assert run["repeat"] == {"max_abs": 0.0, "mean_abs": 0.0}
        errors = [
            float(np.abs(np.asarray(row) - q["bf16_logprobs"]).mean())
            for row, q in zip(run["gold_logprobs"], data["queries"], strict=True)
        ]
        assert np.isfinite(errors).all()
        assert errors == run["query_mae"]
        assert float(np.mean(errors)) == run["mae"]
        assert tool.bootstrap(errors) == run["ci95"]


def test_published_head_probe():
    data = json.loads((ROOT / "docs/data/native-fidelity.json").read_text())[
        "head_probe"
    ]
    ranks = data["by_rank"]
    assert len(ranks) == 2 and ranks[0]["rows"] == ranks[1]["rows"]
    for rank in ranks:
        for key in [
            "rows",
            "global_top20_candidate_misses_on_rank",
            "global_argmax_mismatches",
        ]:
            assert rank[key] == sum(call[key] for call in rank["calls"])
        assert all(1 <= call["rows"] <= 32 for call in rank["calls"])
    assert data["global_top20_tokens"] == ranks[0]["rows"] * 20
    assert data["eligible_rows"] == ranks[0]["rows"]
    assert data["global_top20_retained"] == data["global_top20_tokens"] - sum(
        r["global_top20_candidate_misses_on_rank"] for r in ranks
    )
    assert (
        data["global_top20_recall"]
        == 1
        - sum(r["global_top20_candidate_misses_on_rank"] for r in ranks)
        / data["global_top20_tokens"]
    )
    assert (
        data["final_top1_agreement"]
        == 1 - ranks[0]["global_argmax_mismatches"] / ranks[0]["rows"]
    )


def test_small_batch_persistent_fidelity_data():
    path = ROOT / "docs/data/gdn-m4-fidelity.json"
    if not path.exists():
        pytest.skip("M4 GPU results not collected yet")
    data = json.loads(path.read_text())
    assert data["physical_rows"] == 4
    assert data["query_count"] == len(data["queries"]) == 256
    assert (
        data["target_tokens"]
        == sum(len(q["bf16_logprobs"]) for q in data["queries"])
        == 10479
    )
    tool = module("docs/data/collect_native_comparison.py")
    for arm, run in data["runs"].items():
        errors = [
            float(np.abs(np.asarray(row) - q["bf16_logprobs"]).mean())
            for row, q in zip(run["gold_logprobs"], data["queries"], strict=True)
        ]
        assert run["mae"] == pytest.approx(np.mean(errors))
        assert run["ci95"] == pytest.approx(tool.bootstrap(errors))
        assert run["contract"]["physical_rows"] == 4
        assert run["repeat"] == {"max_abs": 0.0, "mean_abs": 0.0}
    assert all(
        r["prepared_layers"] == 48 and r["persistent_m4"] > 0
        for r in data["runs"]["persistent"]["dispatch"]
    )


def test_published_gdn_equivalence_and_dispatch():
    data = json.loads((ROOT / "docs/data/native-fidelity.json").read_text())
    for arm, reference in (
        ("persistent", "default"),
        ("gdn", "default"),
        ("full_ba", "full"),
        ("full_gdn", "full_ba"),
    ):
        result = data["gdn_ablation"][arm]
        assert result["reference"] == reference
        assert result["exact_gold_logprobs"]
        assert (
            data["runs"][arm]["gold_logprobs"]
            == data["runs"][reference]["gold_logprobs"]
        )
        assert result["mean"] == 0.0 and result["ci95"] == [0.0, 0.0]
        assert {rank["rank"] for rank in result["dispatch"]} == {0, 1}
        for rank in result["dispatch"]:
            assert rank["prepared_layers"] == 48
            if arm in ("gdn", "full_ba", "full_gdn"):
                assert all(rank[f"overlap_m{rows}"] > 0 for rows in (16, 24, 32))
            if arm in ("gdn", "persistent", "full_gdn"):
                assert all(rank[f"persistent_m{rows}"] > 0 for rows in (1, 2, 4, 8))
    small = json.loads((ROOT / "docs/data/gdn-m4-fidelity.json").read_text())
    assert (
        small["runs"]["gdn"]["gold_logprobs"]
        == small["runs"]["persistent"]["gold_logprobs"]
    )
    assert all(rank["persistent_m4"] > 0 for rank in small["runs"]["gdn"]["dispatch"])
