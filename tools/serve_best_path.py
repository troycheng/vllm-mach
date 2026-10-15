#!/usr/bin/env python3
"""Launch a pinned best-profile qualification service, including route receipts."""

import argparse
import json
import os
import sys
from importlib.metadata import distribution
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--sampling", action="store_true")
    a = p.parse_args()
    a.run_dir.mkdir(parents=True)
    # Different pinned runtimes otherwise rebuild the same global JIT cache
    # whenever their include paths differ. Keep compilation out of each
    # other's way; both comparison arms still compile the same runtime code.
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE",
        str(a.run_dir.parents[2] / "flashinfer-workspaces" / Path(sys.prefix).name),
    )
    os.environ["VLLM_MACH_LM_HEAD_SAMPLING"] = str(int(a.sampling))
    os.environ["MACH_BENCH_PROBE_DIR"] = str(a.run_dir / "probe")
    common = {"host": "127.0.0.1", "port": a.port, "served_model_name": "best-path"}
    if a.profile in ("2b-mxfp8", "4b-mxfp8"):
        if a.profile == "2b-mxfp8":
            from vllm_mach.mxfp8.two_b.model import validate_model
            from vllm_mach.mxfp8.two_b.profile import (
                check_environment,
                check_runtime_sources,
            )
            from vllm_mach.mxfp8.two_b.serve import build_argv, configure_environment

            options = {"attention_backend": "TRITON_ATTN", "ssm_dtype": "bfloat16"}
            env = configure_environment(a.run_dir, hybrid_head=True, **options)
        else:
            from vllm_mach.mxfp8.prepare_model import validate_model
            from vllm_mach.mxfp8.profile import check_environment, check_runtime_sources
            from vllm_mach.mxfp8.serve import build_argv, configure_environment

            options = {"ssm_dtype": "bfloat16", "kv_cache_dtype": "fp8_e4m3"}
            env = configure_environment(a.run_dir, **options)
        env["VLLM_MACH_LM_HEAD_SAMPLING"] = str(int(a.sampling))
        os.environ.update(env)
        packages, sources = check_environment(), check_runtime_sources()
        identity = validate_model(a.model, verify_weights=True)
        argv = build_argv(a.model, **common, **options)
    elif a.profile == "4b-block-fp8":
        from vllm_mach.fp8.profile import check_environment, check_runtime_sources
        from vllm_mach.fp8.serve import build_argv, configure_environment

        env = configure_environment(a.run_dir)
        packages, sources = check_environment(), check_runtime_sources()
        identity = {"format": "block-FP8", "existing_checkpoint": True}
        argv = build_argv(a.model, **common)
    else:
        from vllm_mach.mxfp6.install import check_versions, install_profile
        from vllm_mach.mxfp6.serve import build_command

        fp4 = a.profile.endswith("nvfp4")
        moe = a.profile.startswith("35b")
        args = argparse.Namespace(
            model=a.model,
            fp16_ssm=not fp4,
            lossless_prefill=False if fp4 else None,
            owner_prefill=False if fp4 else None,
            nvfp4_lm_head=True,
            verify_prefill=False,
            fused_ar_norm=not fp4,
            fused_ar_quant=not fp4,
            gdn_persistent=not fp4,
            gdn_ba_overlap=not fp4,
        )
        extra = [
            "--host",
            "127.0.0.1",
            "--port",
            str(a.port),
            "--served-model-name",
            "best-path",
            "--kv-cache-memory-bytes",
            str(8 * 2**30),
        ]
        if moe:
            extra += [
                "--max-num-seqs",
                "64",
                "--compilation-config",
                json.dumps(
                    {
                        "mode": "VLLM_COMPILE",
                        "cudagraph_mode": "FULL_AND_PIECEWISE",
                        "cudagraph_capture_sizes": [
                            1,
                            2,
                            4,
                            8,
                            10,
                            12,
                            14,
                            16,
                            20,
                            24,
                            28,
                            32,
                            36,
                            40,
                            48,
                            56,
                            64,
                            72,
                            80,
                            96,
                            112,
                            128,
                            160,
                            192,
                            224,
                            256,
                            320,
                            384,
                            448,
                            512,
                            640,
                            768,
                            784,
                            896,
                            1024,
                            1280,
                            1536,
                            1792,
                            2048,
                        ],
                    }
                ),
            ]
        if fp4:
            extra += [
                "--quantization",
                "compressed-tensors",
                "--mamba-ssm-cache-dtype",
                "float32",
            ]
        command, env = build_command(args, extra)
        # Qualification-only sampling ablation; production uses one hybrid flag.
        env["VLLM_MACH_LM_HEAD_SAMPLING"] = str(int(a.sampling))
        os.environ.update(env)
        os.environ["OMP_NUM_THREADS"] = "4" if moe else "1"
        os.environ["VLLM_MACH_MOE_PROJECTION_TUNING"] = (
            "auto" if moe and not fp4 else "0"
        )
        check_versions()
        sources = install_profile(
            Path(distribution("vllm").locate_file("")),
            Path(distribution("flashinfer-python").locate_file("")),
        )
        if sources["changed_files"]:
            raise RuntimeError("Native source profile is not installed")
        packages = {"vllm": "0.29.0"}
        identity = {"format": "NVFP4" if fp4 else "MXFP6"}
        argv = ["vllm", *command[3:]]
    argv += ["--worker-extension-cls", "serving_route_probe.RouteProbe"]
    selected = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("VLLM_", "MXFP6_", "MX8_"))
        or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MACH_BENCH_PROBE_DIR")
    }
    (a.run_dir / "launch.json").write_text(
        json.dumps(
            {
                "profile": a.profile,
                "argv": argv,
                "environment": selected,
                "packages": packages,
                "sources": sources,
                "model": identity,
            },
            indent=2,
        )
        + "\n"
    )
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = argv
    vllm_main()


if __name__ == "__main__":
    main()
