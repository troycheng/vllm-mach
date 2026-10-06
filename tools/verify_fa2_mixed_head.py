# SPDX-License-Identifier: Apache-2.0
"""Diagnose #58013 exact-source behavior with public synthetic FA2 inputs.

This is exact-source operand qualification with runtime compatibility bindings,
not execution of a complete upstream-head installation or model evaluation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from urllib.request import urlopen

BASE = "382970ee6ca490aeaaaf4e32c53695b581ff61ba"
HEAD = "e17555d2673e632cbe85365a53490a138e4b2fae"
PUBLIC_SOURCES = {
    "base_flash_attn.py": (
        f"https://raw.githubusercontent.com/vllm-project/vllm/{BASE}/vllm/v1/attention/backends/flash_attn.py",
        "0598232cb726e8ce97ec1db725a6558218e51b7a8051abd7d74193d2283cacb1"),
    "head_flash_attn.py": (
        f"https://raw.githubusercontent.com/LiRunGuo/vllm/{HEAD}/vllm/v1/attention/backends/flash_attn.py",
        "75aef45298c454532cb3dd59954bea80b06498db82d8664b3c1718b42313574c"),
    "head_utils.py": (
        f"https://raw.githubusercontent.com/LiRunGuo/vllm/{HEAD}/vllm/v1/attention/backends/utils.py",
        "ff92a61b8a0439539cc8cf9dd353efc29765d8094a4948b4824188de0bde9063"),
}


def public_sources(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in PUBLIC_SOURCES.items():
        path = directory / name
        if not path.is_file():
            with urlopen(url, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Official source fingerprint differs: {name}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source cache fingerprint differs: {name}")
    return {"repository": "https://github.com/LiRunGuo/vllm", "base": BASE,
            "head": HEAD, "base_sha256": PUBLIC_SOURCES["base_flash_attn.py"][1],
            "head_sha256": PUBLIC_SOURCES["head_flash_attn.py"][1],
            "helper_sha256": PUBLIC_SOURCES["head_utils.py"][1],
            "original_author": "LiRunGuo"}


def synthetic_case(torch, case, seed):
    """New diagnostic data; no captured requests, weights or original operands."""
    torch.manual_seed(seed + case)
    nd, suffix = (30, (526, 1492)) if case == 32 else (60, (646, 1342))
    lengths = [1] * nd + list(suffix)
    actual, n, blocks, page = sum(lengths), len(lengths), 64, 528
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                          dtype=torch.int32)
    query = torch.randn((actual, 16, 256), dtype=torch.bfloat16)
    cache = torch.empty_strided((blocks, 4, page, 512),
                               (4 * page * 512, 512, 4 * 512, 1),
                               dtype=torch.bfloat16).normal_()
    source = {"num_actual_tokens": actual, "decode_reqs": nd,
              "prefill_reqs": n - nd, "max_query_len": max(lengths),
              "max_seq_len": 3072, "scale": 0.0625, "fa_version": 2,
              "num_splits": 0, "query_stride": list(query.stride()),
              "cache_stride": list(cache.stride())}
    return {"query": query, "cache": cache, "source": source,
            "query_start_loc": starts,
            "seq_lens": torch.full((n,), 3072, dtype=torch.int32),
            "block_table": torch.randint(0, blocks, (n, 6), dtype=torch.int32)}


def load_methods(path, runtime, record_fa, helper_tree):
    tree = ast.parse(path.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    metadata = classes["FlashAttentionMetadata"]
    build = copy.deepcopy(next(node for node in classes["FlashAttentionMetadataBuilder"].body
                               if isinstance(node, ast.FunctionDef) and node.name == "build"))
    forward = copy.deepcopy(next(node for node in classes["FlashAttentionImpl"].body
                                 if isinstance(node, ast.FunctionDef) and node.name == "forward"))
    wrapper = copy.deepcopy(classes["FA4DenseAttentionKernel"])
    wrapper.bases = []
    wrapper.body = [node for node in wrapper.body if isinstance(node, ast.FunctionDef)
                    and node.name in ("kernel", "__call__")]
    helper = copy.deepcopy(next(node for node in helper_tree.body
                                if isinstance(node, ast.FunctionDef)
                                and node.name == "split_decodes_and_prefills"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                            level=0), metadata, wrapper, helper, build, forward]
    namespace = dict(runtime.__dict__)
    namespace["flash_attn_varlen_func"] = record_fa
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), namespace)
    namespace["_FA4_DENSE_ATTENTION_KERNEL"] = namespace["FA4DenseAttentionKernel"]()
    return namespace["build"], namespace["forward"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path,
                        help="Optional user-owned captured fixtures; never required")
    parser.add_argument("--sources", type=Path,
                        help="Official-source cache; fetched from pinned public URLs if absent")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends import flash_attn as runtime

    if importlib.metadata.version("vllm") != "0.29.0":
        raise RuntimeError("This compatibility driver requires vLLM 0.29.0")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError("This compatibility driver requires Torch 2.13.0")
    if not torch.cuda.is_available():
        raise RuntimeError("This FA2 diagnostic requires a CUDA GPU")
    torch.set_num_threads(1)
    args.sources = args.sources or args.output.parent / "fa2-upstream-sources"
    sources = public_sources(args.sources)
    for kind in ("base", "head"):
        assert hashlib.sha256((args.sources / f"{kind}_flash_attn.py").read_bytes()
                              ).hexdigest() == sources[f"{kind}_sha256"]
    helper_source = args.sources / "head_utils.py"
    helper_sha = hashlib.sha256(helper_source.read_bytes()).hexdigest()
    assert helper_sha == sources["helper_sha256"]
    helper_tree = ast.parse(helper_source.read_text())
    real_fa = runtime.flash_attn_varlen_func
    calls = []
    force_head_splits_one = False

    def recorder(*positional, **kwargs):
        requested = kwargs["num_splits"]
        effective = 1 if force_head_splits_one else requested
        calls.append({"tokens": kwargs["q"].shape[0],
                      "qmax": kwargs["max_seqlen_q"],
                      "requested_num_splits": requested,
                      "effective_num_splits": effective,
                      "diagnostic_override": force_head_splits_one})
        if force_head_splits_one:
            kwargs = {**kwargs, "num_splits": effective}
        return real_fa(*positional, **kwargs)

    base_build, base_forward = load_methods(args.sources / "base_flash_attn.py",
                                            runtime, recorder, helper_tree)
    head_build, head_forward = load_methods(args.sources / "head_flash_attn.py",
                                            runtime, recorder, helper_tree)

    def builder(method, split):
        obj = object.__new__(runtime.FlashAttentionMetadataBuilder)
        obj.vllm_config = NS(parallel_config=NS(tensor_parallel_size=1,
            pipeline_parallel_size=1, enable_dbo=False), speculative_config=None)
        obj.dcp_world_size = 1
        obj.aot_schedule = False
        obj.aot_sliding_window = (-1, -1)
        obj.use_full_cuda_graph = False
        obj.max_cudagraph_size = None
        obj.max_num_splits = 0
        obj.mm_prefix_query_ranges_np = None
        obj.rswa_window = None
        obj.split_mixed_batch = split
        obj.build = MethodType(method, obj)
        return obj

    base_builder, head_builder = builder(base_build, False), builder(head_build, True)
    fixture_manifest = (json.loads((args.fixtures / "manifest.json").read_text())
                        if args.fixtures is not None else None)
    rows = []
    for case in (32, 64):
        if fixture_manifest is None:
            digest = None
            data = synthetic_case(torch, case, args.seed)
        else:
            identity = fixture_manifest[str(case)]
            path = args.fixtures / identity["file"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == identity["sha256"]
            data = torch.load(path, map_location="cpu", weights_only=True)
        source = data["source"]
        actual, nd = source["num_actual_tokens"], source["decode_reqs"]
        q = torch.empty_strided((actual + 3, 16, 256), source["query_stride"],
                               device="cuda", dtype=torch.bfloat16)
        q[:actual].copy_(data["query"])
        q[actual:].zero_()
        cache = torch.empty_strided(data["cache"].shape, source["cache_stride"],
                                   device="cuda", dtype=torch.bfloat16)
        cache.copy_(data["cache"])
        out = torch.full_like(q, -9)
        expected = (data["output"].cuda().reshape(actual, 16, 256)
                    if "output" in data else None)
        n = data["seq_lens"].numel()
        common = CommonAttentionMetadata(
            query_start_loc=data["query_start_loc"].cuda(),
            query_start_loc_cpu=data["query_start_loc"],
            seq_lens=data["seq_lens"].cuda(), num_reqs=n, num_actual_tokens=actual,
            max_query_len=source["max_query_len"], max_seq_len=source["max_seq_len"],
            block_table_tensor=data["block_table"].cuda(),
            slot_mapping=torch.full((actual,), -1, device="cuda", dtype=torch.int64),
            seq_lens_cpu_upper_bound=data["seq_lens"])
        impl = runtime.FlashAttentionImpl(num_heads=16, head_size=256,
            scale=source["scale"], num_kv_heads=4, alibi_slopes=None,
            sliding_window=None, kv_cache_dtype="auto")
        assert impl.vllm_flash_attn_version == 2
        layer = NS(_q_scale=torch.ones((), device="cuda"),
                   _k_scale=torch.ones((), device="cuda"),
                   _v_scale=torch.ones((), device="cuda"))

        def check(label, cmd, qview, outview, expected_calls, saved=None,
                  *, diagnostic_force_splits_one=False):
            nonlocal force_head_splits_one
            original_meta = base_builder.build(0, cmd)
            head_meta = head_builder.build(0, cmd)
            calls.clear()
            force_head_splits_one = False
            outview.fill_(-9)
            base_forward(impl, layer, qview, qview[:, :4], qview[:, :4], cache,
                         original_meta, outview)
            assert len(calls) == 1
            base_calls = list(calls)
            reference = outview[:cmd.num_actual_tokens].clone()
            if saved is not None:
                assert torch.equal(reference.view(torch.int16), saved.view(torch.int16))
            calls.clear()
            outview.fill_(-9)
            force_head_splits_one = diagnostic_force_splits_one
            try:
                result = head_forward(impl, layer, qview, qview[:, :4], qview[:, :4],
                                      cache, head_meta, outview)
            finally:
                force_head_splits_one = False
            torch.cuda.synchronize()
            assert result is outview and len(calls) == expected_calls
            changed = outview[:cmd.num_actual_tokens].view(torch.int16) != reference.view(torch.int16)
            count = int(changed.sum())
            if expected_calls == 1:
                assert count == 0, (label, "single-call control differs", count)
            assert bool((outview[cmd.num_actual_tokens:] == -9).all())
            delta = (outview[:cmd.num_actual_tokens].float() - reference.float()).abs()
            assert bool(torch.isfinite(delta).all())
            split_nd = head_meta.num_split_decodes
            return {"label": label, "base_calls": base_calls, "calls": list(calls),
                    "changed": count, "byte_exact": count == 0,
                    "decode_changed": int(changed[:split_nd].sum()),
                    "prefill_changed": int(changed[split_nd:].sum()),
                    "mae": float(delta.mean()), "max_abs": float(delta.max()),
                    "num_split_decodes": split_nd,
                    "source_unchanged": True,
                    "is_original_head_execution": not diagnostic_force_splits_one,
                    "diagnostic": ("head FA2 calls force num_splits=1; base retains 0"
                                   if diagnostic_force_splits_one else None)}

        results = [check("mixed_input", common, q, out, 2, expected)]
        assert results[0]["num_split_decodes"] == nd
        diagnostic = check("mixed_input_head_forced_splits_one", common, q, out,
                           2, expected, diagnostic_force_splits_one=True)
        for label, reqs, tokens, shift, qmax in (
            ("pure_decode", slice(0, nd), slice(0, nd + 3), 0, 1),
            ("pure_prefill", slice(nd, None), slice(nd, None), nd, source["max_query_len"]),
        ):
            control = copy.copy(common)
            start = 0 if label == "pure_decode" else nd
            stop = nd if label == "pure_decode" else n
            control.query_start_loc_cpu = data["query_start_loc"][start:stop + 1] - shift
            control.query_start_loc = control.query_start_loc_cpu.cuda()
            control.seq_lens = common.seq_lens[reqs]
            control.block_table_tensor = common.block_table_tensor[reqs]
            control.seq_lens_cpu_upper_bound = data["seq_lens"][reqs]
            control.num_reqs = stop - start
            control.num_actual_tokens = nd if label == "pure_decode" else actual - nd
            control.max_query_len = qmax
            control.slot_mapping = common.slot_mapping[:control.num_actual_tokens]
            results.append(check(label, control, q[tokens], out[tokens], 1))
        rows.append({"case": case, "fixture_sha256": digest,
                     "query_lengths": data["query_start_loc"].diff().tolist(),
                     "max_seq_len": source["max_seq_len"], "checks": results,
                     "diagnostic_head_forced_splits_one": diagnostic})
        print(json.dumps(rows[-1]), flush=True)
        del q, cache, out, expected, common, control, data
        torch.cuda.empty_cache()
    report = {"status": "COMPLETE", "upstream_head": sources["head"],
              "scope": "unchanged exact-head/base AST methods on vLLM 0.29 FA2 runtime",
              "torch": torch.__version__, "sources": sources,
              "input_kind": "synthetic" if fixture_manifest is None else "user_owned_capture",
              "synthetic_seed": args.seed if fixture_manifest is None else None,
              "cuda_device": torch.cuda.get_device_name(),
              "compute_capability": list(torch.cuda.get_device_capability()),
              "runtime_source_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
              "runtime_compatibility_bindings": ["stock vLLM globals and FA2 binary",
                  "unchanged upstream FA4DenseAttentionKernel.kernel/__call__",
                  "model-independent builder state; constructor not exercised"],
              "full_upstream_runtime_qualified": False,
              "mixed_graph_forced": False, "model_or_service_qualified": False,
              "original_head_mixed_byte_exact": all(
                  row["checks"][0]["byte_exact"] for row in rows),
              "diagnostic_head_forced_splits_one_byte_exact": all(
                  row["diagnostic_head_forced_splits_one"]["byte_exact"] for row in rows),
              "cause_attribution": "No cause inferred; compare recorded operands and results",
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
