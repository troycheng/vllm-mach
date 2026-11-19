"""Matched request content and arrival schedule are independent of model arm."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

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


@pytest.mark.parametrize("release", ["0.33.0", "0.34.0"])
def test_native_launcher_only_sets_runner_flag_on_vllm_033(monkeypatch, release):
    from vllm_mach.mxfp6 import serve

    monkeypatch.setattr(serve, "BASE_PACKAGES", {"vllm": release})
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    args = argparse.Namespace(
        model="/models/dense", lossless_prefill=None, owner_prefill=None,
        fp16_ssm=False, verify_prefill=False,
    )
    _, env = serve.build_command(args, [])
    if release == "0.33.0":
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    else:
        assert "VLLM_USE_V2_MODEL_RUNNER" not in env
    in_process_env = {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    serve.update_profile_environment(in_process_env, env)
    assert in_process_env.get("VLLM_USE_V2_MODEL_RUNNER") == env.get("VLLM_USE_V2_MODEL_RUNNER")
    assert env["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


def test_manifest_must_cover_all_requests(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"prompts": [{"token_ids": [1, 2]}]}))
    with pytest.raises(AssertionError):
        load().make_contract(args(prompt_manifest=path))


@pytest.mark.parametrize("release", ["0.33.0", "0.34.0"])
@pytest.mark.parametrize("arm", ["fp8", "nvfp4", "core_stock_aligned"])
def test_stock_baseline_keeps_compilation_and_disables_flashinfer_ar(
    monkeypatch, arm, release
):
    from vllm_mach.mxfp6 import serve

    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(serve, "BASE_PACKAGES", {"vllm": release})
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "1")
    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "1")
    config = argparse.Namespace(
        models=Path("/models"),
        stock_runtime=Path("/stock"),
        devices="0,1",
        port=8000,
    )
    command, env = tool.launch_configuration(config, arm)
    assert "--compilation-config" not in command
    assert "--kv-cache-memory-bytes" not in command
    assert "--block-size" not in command
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert env["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
    if arm in tool.STOCK_ARMS:
        assert env["VLLM_PLUGINS"] == ""
        assert "VLLM_QWEN3_5_FUSED_AR_NORM" not in env
        assert env["PYTHONPATH"] == "/stock"
    else:
        assert env["VLLM_PLUGINS"] == "mach"
        assert env["VLLM_QWEN3_5_FUSED_AR_NORM"] == "0"
    native, _ = tool.launch_configuration(config, "default")
    if release == "0.34.0":
        assert native[native.index("--block-size") + 1] == "16"
    else:
        assert "--block-size" not in native


@pytest.mark.parametrize("release", ["0.33.0", "0.34.0"])
@pytest.mark.parametrize("profile", ["27b-mxfp6", "35b-mxfp6", "27b-nvfp4", "35b-nvfp4"])
def test_best_path_uses_selected_source_tree_and_cache_geometry(tmp_path, release, profile):
    older_baseline = release == "0.33.0"
    src = ROOT / "src"
    if older_baseline:
        src = tmp_path / "baseline" / "src"
        package = src / "vllm_mach" / "mxfp6"
        package.mkdir(parents=True)
        for directory in (package.parent, package):
            (directory / "__init__.py").touch()
        (package / "install.py").write_text(dedent('''
            def check_versions(): pass
            def install_profile(*args): pass
        '''))
        (package / "serve.py").write_text(dedent('''
            import os

            def build_command(args, extra, /):
                env = dict(os.environ)
                env.pop("VLLM_MACH_MOE_HIGH_M_COMPACT", None)
                env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
                env["MACH_TEST_BASELINE"] = "1"
                return ["python", "-m", "vllm", "serve", str(args.model), *extra], env
        '''))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe" if profile.startswith("35b") else "qwen3_5",
    }))
    runtime = tmp_path / "runtime"
    # Stub runtime-only imports while letting Python select and load the
    # production or older Mach source package through PYTHONPATH.
    harness = dedent('''
        import json, os, runpy, sys
        from types import SimpleNamespace
        from unittest.mock import patch

        release = sys.argv.pop(1)
        sys.argv.pop(0)
        def launch():
            print(json.dumps({"argv": sys.argv, "environment": {key: os.environ.get(key)
                for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_MACH_MOE_HIGH_M_COMPACT",
                            "MACH_TEST_BASELINE", "MACH_TEST_PRESERVED")}}))

        sys.modules["vllm.entrypoints.cli.main"] = SimpleNamespace(main=launch)
        with patch("importlib.metadata.version", return_value=release), \
             patch("importlib.metadata.distribution") as distribution:
            distribution.return_value.locate_file.return_value = "/unused"
            from vllm_mach.mxfp6 import install
            install.check_versions = lambda: None
            install.install_profile = lambda *args: {"changed_files": []}
            runpy.run_path(sys.argv[0], run_name="__main__")
    ''')
    result = subprocess.run(
        [sys.executable, "-c", harness, release, str(ROOT / "tools/serve_best_path.py"),
         "--profile", profile, "--model", str(tmp_path), "--run-dir", str(runtime),
         "--port", "8000"],
        env=dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1",
                 VLLM_USE_V2_MODEL_RUNNER="0", VLLM_MACH_MOE_HIGH_M_COMPACT="1",
                 MACH_TEST_PRESERVED="keep"),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["environment"] == {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if older_baseline else None,
        "VLLM_MACH_MOE_HIGH_M_COMPACT": None,
        "MACH_TEST_BASELINE": "1" if older_baseline else None,
        "MACH_TEST_PRESERVED": "keep",
    }
    argv = observed["argv"]
    if older_baseline or profile.endswith("nvfp4"):
        assert "--block-size" not in argv
    else:
        assert argv[argv.index("--block-size") + 1] == "16"
    assert json.loads((runtime / "launch.json").read_text())["argv"] == argv


def test_published_serving_counts_and_throughput():
    data = json.loads((ROOT / "docs/data/native-serving.json").read_text())
    assert set(data["runs"]) == {
        "fp8",
        "nvfp4",
        "default",
        "persistent",
        "gdn",
        "full",
        "full_ba",
        "full_gdn",
        "prefill_default",
    }
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


@pytest.mark.parametrize(
    "arm,base,changed",
    [
        ("persistent", "default", "VLLM_MACH_GDN_PERSISTENT"),
        ("full_ba", "full", "VLLM_MACH_GDN_BA_OVERLAP"),
        ("full_gdn", "full_ba", "VLLM_MACH_GDN_PERSISTENT"),
        ("dense_lossless", "dense_none", "VLLM_SM120_LOSSLESS_PREFILL"),
        ("dense_owner", "dense_none", "VLLM_SM120_OWNER_PREFILL"),
        ("dense_default", "dense_lossless", "VLLM_SM120_OWNER_PREFILL"),
        ("dense_default", "dense_owner", "VLLM_SM120_LOSSLESS_PREFILL"),
        ("dense_head", "dense_default", "VLLM_HYBRID_NVFP4_LM_HEAD"),
        ("dense_full", "dense_ssm", "VLLM_HYBRID_NVFP4_LM_HEAD"),
        ("core_sampler", "core_graph", "VLLM_VOCAB_PARALLEL_GREEDY"),
        ("core_ar_norm", "core_sampler", "VLLM_QWEN3_5_FUSED_AR_NORM"),
        ("core_gdn", "core_ar_norm", "VLLM_MACH_GDN_PERSISTENT"),
        ("core_ba", "core_gdn", "VLLM_MACH_GDN_BA_OVERLAP"),
        ("core_swiglu", "core_ba", "VLLM_MACH_FUSED_SWIGLU_QUANT"),
        ("core_gdn_quant", "core_swiglu", "VLLM_MACH_FUSED_GDN_QUANT"),
        ("core_ar_quant", "core_gdn_quant", "VLLM_MACH_FUSED_AR_QUANT"),
    ],
)
def test_gdn_serving_ablation_changes_only_expected_flags(arm, base, changed):
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = argparse.Namespace(
        models=Path("/models"), stock_runtime=Path("/stock"), devices="0,1", port=8000
    )
    command, env = tool.launch_configuration(args, arm)
    base_command, base_env = tool.launch_configuration(args, base)
    assert command == base_command
    expected = {changed}
    if changed == "VLLM_HYBRID_NVFP4_LM_HEAD":
        expected.add("VLLM_MACH_LM_HEAD_SAMPLING")
    assert {k for k in env if env[k] != base_env[k]} == expected
    for key in expected:
        assert env[key] == "1" and base_env[key] == "0"


@pytest.mark.parametrize("arm,base", [("dense_ssm", "dense_default"), ("dense_full", "dense_head")])
def test_dense_ssm_ablation_changes_state_allocation_and_admission(arm, base):
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = argparse.Namespace(
        models=Path("/models"), stock_runtime=Path("/stock"), devices="0,1", port=8000
    )
    command, env = tool.launch_configuration(args, arm)
    base_command, base_env = tool.launch_configuration(args, base)
    index = command.index("--mamba-ssm-cache-dtype")
    assert command[index + 1] == "float16"
    assert command[:index] + command[index + 2:] == base_command
    assert {k for k in env if env[k] != base_env[k]} == {"VLLM_QWEN3_5_FP16_SSM"}
    assert env["VLLM_QWEN3_5_FP16_SSM"] == "1"
    assert base_env["VLLM_QWEN3_5_FP16_SSM"] == "0"


def test_core_final_reconstructs_base_profile():
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = argparse.Namespace(
        models=Path("/models"), stock_runtime=Path("/stock"), devices="0,1", port=8000
    )
    final, final_env = tool.launch_configuration(args, "core_ar_quant")
    base, base_env = tool.launch_configuration(args, "dense_none")
    assert final == base
    for key in ("VLLM_MACH_FUSED_SWIGLU_QUANT", "VLLM_MACH_FUSED_GDN_QUANT"):
        assert final_env.pop(key) == "1"
        base_env.pop(key, None)
    assert final_env == base_env


def test_stock_aligned_profile_restores_defaults_without_mach_optimizations():
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = argparse.Namespace(
        models=Path("/models"), stock_runtime=Path("/stock"), devices="2,3", port=8000
    )
    command, env = tool.launch_configuration(args, tool.ALIGNED_ARM)
    _, control_env = tool.launch_configuration(args, "core_graph")
    for key, value in control_env.items():
        if key in env and key.startswith(("VLLM_MACH_", "VLLM_QWEN3_5_", "VLLM_SM120_")):
            assert env[key] == value
    for flag in ("--compilation-config", "--attention-backend",
                 "--max-num-batched-tokens", "--kv-cache-memory-bytes", "--enforce-eager"):
        assert flag not in command
    for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_BREAKABLE_CUDAGRAPH",
                "VLLM_FLASHINFER_ALLREDUCE_BACKEND"):
        assert key not in env
    assert env["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.9"
    assert env["VLLM_VOCAB_PARALLEL_GREEDY"] == "0"
    assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == "0"


def test_core_stages_all_keep_decode_graphs():
    spec = importlib.util.spec_from_file_location(
        "serving_comparison", ROOT / "tools/compare_native_serving.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = argparse.Namespace(
        models=Path("/models"), stock_runtime=Path("/stock"), devices="0,1", port=8000
    )
    assert all("eager" not in arm for arm in tool.DENSE_CORE_ORDER)
    for arm in tool.DENSE_CORE_ORDER:
        command, _ = tool.launch_configuration(args, arm)
        assert "--enforce-eager" not in command
        cfg = json.loads(command[command.index("--compilation-config") + 1])
        assert cfg["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert cfg["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 24, 32]


@pytest.mark.parametrize("token_text", ["hello", "", "hello\u2028world\u0085end"])
def test_single_token_prefill_benchmark_serializes_without_decode_intervals(token_text):
    import asyncio

    from aiohttp import web

    tool = load()

    async def exercise():
        async def completion(request):
            payload = await request.json()
            events = [
                {"choices": [{"text": token_text, "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(payload["prompt"]),
                        "completion_tokens": 1,
                    },
                },
            ]
            body = "".join(
                "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                for event in events
            )
            return web.Response(
                text=body + "data: [DONE]\n\n", content_type="text/event-stream"
            )

        app = web.Application()
        app.router.add_post("/v1/completions", completion)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            result = await tool.run(
                args(
                    base_url=f"http://127.0.0.1:{port}",
                    model="test",
                    timeout_s=10,
                    prompt_manifest=None,
                    input_tokens=4,
                    output_tokens=1,
                    num_prompts=2,
                    request_rate=0,
                    warmup_requests=1,
                    warmup_output_tokens=1,
                )
            )
        finally:
            await runner.cleanup()
        json.dumps(result, allow_nan=False)
        assert result["aggregate"]["completed"] == 2
        assert result["aggregate"]["mean_ttft_ms"] >= 0
        for metric in ("mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms"):
            assert result["aggregate"][metric] is None

    asyncio.run(exercise())
