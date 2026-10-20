"""Current packaged 2B/4B fixed-gold validation; reuse the published scorer."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

MACH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MACH / "tools"), str(MACH / "tools/quantization")]


def write(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def receipt(wrapper, profile):
    worker = getattr(wrapper, "worker", wrapper)
    if profile == "2b":
        from vllm_mach.mxfp8.two_b.profile import inspect_worker
        return inspect_worker(worker)
    from vllm_mach.mxfp8.profile import write_receipt
    return write_receipt(worker, "refresh_quality")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("2b", "4b"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=MACH / "docs/data/fidelity-samples.json")
    p.add_argument("--rows", type=int, choices=(32, 64), required=True)
    p.add_argument("--ssm-dtype", choices=("float32", "bfloat16"), required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=False)
    manifest = a.manifest
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert manifest_sha == "f51cdf0fec03cf4b66f7391afcccf397dd6418509edafc2c7d1d222f9db29cf2"
    samples = json.loads(manifest.read_text())["samples"]
    assert len(samples) == 256
    assert sum(len(s["target_token_ids"]) for s in samples) == 10479
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1", K4O_QUALITY_ARM="m32",
        VLLM_SMALL_BATCH_ATTN="0", VLLM_MACH_MXFP8_BF16_GDN_GROUPED="0",
        VLLM_MACH_MXFP8_BF16_GDN_PDL="0",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(MACH / "tools"), str(MACH / "tools/quantization"),
         os.environ.get("PYTHONPATH", "")]
    )
    model = a.model.resolve()
    if a.profile == "2b":
        from vllm_mach.mxfp8.two_b.serve import configure_environment
        from vllm_mach.mxfp8.two_b.profile import quality_contract, check_environment, check_runtime_sources
        from vllm_mach.mxfp8.two_b.model import validate_model
        configure_environment(a.outdir / "run", quality_rows=a.rows,
                              attention_backend="TRITON_ATTN", hybrid_head=True,
                              ssm_dtype=a.ssm_dtype)
        config = dict(dtype="bfloat16", tensor_parallel_size=1,
                      enable_chunked_prefill=True, enable_prefix_caching=False,
                      attention_backend="TRITON_ATTN", language_model_only=True,
                      seed=20260907, logprobs_mode="raw_logprobs", disable_log_stats=True,
                      mamba_ssm_cache_dtype=a.ssm_dtype, kv_cache_dtype="bfloat16",
                      quantization="compressed-tensors",
                      kernel_config={"linear_backend": "flashinfer_cutlass"},
                      **quality_contract(a.rows))
    else:
        from vllm_mach.mxfp8.serve import configure_environment
        from vllm_mach.mxfp8.profile import quality_contract, check_environment, check_runtime_sources
        from vllm_mach.mxfp8.prepare_model import validate_model
        configure_environment(a.outdir / "run", quality_rows=a.rows,
                              ssm_dtype=a.ssm_dtype, kv_cache_dtype="fp8_e4m3")
        config = quality_contract(a.rows)
    kwargs = dict(model=str(model), tokenizer=str(model), **config)
    from measure_mxfp8_kv_quality import install_teacher, score
    import torch
    from vllm import LLM
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(a.outdir / "contract.json", dict(
        mach_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=MACH, capture_output=True,
            text=True, check=False).stdout.strip() or None,
        native_library_environment={key: os.environ.get(key) for key in
                                    ("MXFP8_LIBRARY_PATH", "MXFP6_LIBRARY_PATH")},
        profile=a.profile, rows=a.rows, ssm_dtype=a.ssm_dtype, llm_args=kwargs,
        manifest_sha256=manifest_sha,
        teacher_sha256=hashlib.sha256((MACH / "tools/quantization/teacher_decode.py").read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256((MACH / "tools/measure_mxfp8_kv_quality.py").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=validate_model(model, verify_weights=True), versions=check_environment(),
        runtime_sources=check_runtime_sources()))
    started = time.time()
    engine = LLM(**kwargs)
    try:
        write(a.outdir / "runtime.json", engine.collective_rpc(
            functools.partial(receipt, profile=a.profile)))
        write(a.outdir / "teacher.json", engine.collective_rpc(
            functools.partial(install_teacher, rows=a.rows)))
        records, repeat = score(engine, samples, a.rows, config["max_num_batched_tokens"])
        write(a.outdir / "records.json", records)
        write(a.outdir / "final-runtime.json", engine.collective_rpc(
            functools.partial(receipt, profile=a.profile)))
        write(a.outdir / "complete.json", dict(
            complete=True, queries=len(records),
            gold_tokens=sum(len(r["gold_logprobs"]) for r in records),
            repeat=repeat, elapsed_s=time.time()-started))
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
