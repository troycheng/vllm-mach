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


def test_current_profile_fidelity():
    path = ROOT / "docs/data/profile-fidelity-20260917.json"
    assert path.is_file()
    import hashlib

    data = json.loads(path.read_text())
    assert (
        data["manifest_sha256"]
        == hashlib.sha256(
            (ROOT / "docs/data/fidelity-samples.json").read_bytes()
        ).hexdigest()
    )
    assert data["query_count"] == len(data["queries"]) == 256
    bootstrap = module("docs/data/collect_native_comparison.py").bootstrap
    for family in ("dense", "moe"):
        for rows in (4, 32):
            result = data["models"][family][f"m{rows}"]
            runs = result["runs"]
            assert set(runs) == {"bf16", "default", "full", "fp8", "nvfp4"}
            reference = runs["bf16"]["gold_logprobs"]
            assert sum(map(len, reference)) == data["target_tokens"] == 10479
            checked_runs = dict(runs)
            if family == "moe" and rows == 32:
                checked_runs["nvfp4_repeat"] = data["moe_nvfp4_m32_independent_repeat"]
            for arm, run in checked_runs.items():
                errors = [
                    float(np.abs(np.asarray(row) - ref).mean())
                    for row, ref in zip(run["gold_logprobs"], reference, strict=True)
                ]
                assert len(errors) == 256
                assert errors == run["query_mae"]
                assert float(np.mean(errors)) == run["mae"]
                assert bootstrap(errors) == run["ci95"]
                repeated = run["repeat_gold_logprobs"]
                assert len(repeated) == rows
                delta = [
                    abs(x - y)
                    for first, second in zip(
                        run["gold_logprobs"][:rows], repeated, strict=True
                    )
                    for x, y in zip(first, second, strict=True)
                ]
                assert run["repeat"] == {
                    "max_abs": max(delta),
                    "mean_abs": sum(delta) / len(delta),
                }
                audit = run["physical_batch_audit"]
                assert audit["physical_rows"] == rows
                assert audit["cohorts_including_repeat"] == 256 // rows + 1
                assert (
                    audit["scored_decode_steps_per_rank"]["0"]
                    == audit["scored_decode_steps_per_rank"]["1"]
                    > 0
                )
                contract = run["contract"]
                assert contract["physical_rows"] == rows
                assert contract["profile_version"] == "current"
                env = contract["environment"]
                mach = arm in ("default", "full")
                assert env["VLLM_PLUGINS"] == ("mach" if mach else "")
                assert env["VLLM_SM120_OWNER_PREFILL"] == str(
                    int(mach and family == "dense")
                )
                assert env["VLLM_SM120_LOSSLESS_PREFILL"] == str(
                    int(mach and family == "dense")
                )
                assert env["VLLM_QWEN3_5_FP16_SSM"] == str(int(arm == "full"))
                assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == str(int(arm == "full"))
                assert env["VLLM_MACH_GDN_PERSISTENT"] == str(int(mach))
                assert env["VLLM_MACH_GDN_BA_OVERLAP"] == str(int(mach))
                if mach:
                    assert {r["rank"] for r in run["dispatch"]} == {0, 1}
                    for rank in run["dispatch"]:
                        assert rank["prepared_layers"] == (
                            48 if family == "dense" else 30
                        )
                        if family == "dense":
                            assert (
                                rank["persistent_m4" if rows == 4 else "overlap_m32"]
                                > 0
                            )
                        else:
                            # Current MoE uses the compiled native GDN path; flags
                            # and prepared layers do not prove fast-path dispatch.
                            assert (
                                "mode" not in contract["llm_args"]["compilation_config"]
                            )
            paired = (
                np.asarray(runs["full"]["query_mae"]) - runs["default"]["query_mae"]
            )
            assert result["full_minus_default"] == dict(
                mean=float(paired.mean()), ci95=bootstrap(paired)
            )


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe"])
def test_fidelity_uses_current_model_aware_profiles(tmp_path, family):
    tool = module("tools/fidelity_native_mxfp6.py")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family}))
    default = tool.profile_flags("default", tmp_path, current_profile=True)
    full = tool.profile_flags("full", tmp_path, current_profile=True)
    assert {key for key in default if default[key] != full[key]} == {
        "VLLM_QWEN3_5_FP16_SSM",
        "VLLM_HYBRID_NVFP4_LM_HEAD",
    }
    for flags in (default, full):
        assert flags["VLLM_MACH_GDN_PERSISTENT"] == "1"
        assert flags["VLLM_MACH_GDN_BA_OVERLAP"] == "1"
        assert flags["VLLM_SM120_LOSSLESS_PREFILL"] == str(int(family == "qwen3_5"))
        assert flags["VLLM_SM120_OWNER_PREFILL"] == str(int(family == "qwen3_5"))
    for arm in ("bf16", "fp8", "nvfp4"):
        flags = tool.profile_flags(arm, tmp_path, current_profile=True)
        assert flags["VLLM_PLUGINS"] == ""
        assert flags["VLLM_SM120_OWNER_PREFILL"] == "0"
        assert flags["VLLM_SM120_LOSSLESS_PREFILL"] == "0"
        assert flags["VLLM_QWEN3_5_FP16_SSM"] == "0"
        assert flags["VLLM_MACH_GDN_PERSISTENT"] == "0"
        assert flags["VLLM_MACH_GDN_BA_OVERLAP"] == "0"
    if family == "qwen3_5":
        historical = tool.profile_flags("default", tmp_path)
        assert historical["VLLM_MACH_GDN_PERSISTENT"] == "0"
        assert historical["VLLM_SM120_OWNER_PREFILL"] == "0"
