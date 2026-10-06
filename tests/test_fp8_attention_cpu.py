"""CPU metadata and normal-forward operands; these do not validate FA2 math."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
from importlib import metadata as package_metadata
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def attention(monkeypatch):
    path = REPO / "src/vllm_mach/fp8/attention.py"
    spec = importlib.util.spec_from_file_location("mach_fp8_attention_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_MACH_FP8_FA2", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return module


def config():
    return NS(parallel_config=NS(tensor_parallel_size=1,
                                pipeline_parallel_size=1, enable_dbo=False),
              speculative_config=None)


def metadata(lengths):
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                           dtype=torch.int32)
    common = NS(num_reqs=len(lengths), query_start_loc_cpu=offsets,
                query_start_loc=offsets)
    meta = NS(num_actual_tokens=sum(lengths), max_query_len=max(lengths),
              max_seq_len=8192, query_start_loc=offsets,
              seq_lens=torch.arange(len(lengths), dtype=torch.int32) + 4096,
              block_table=torch.arange(len(lengths) * 16, dtype=torch.int32)
              .reshape(len(lengths), 16),
              use_cascade=False, common_prefix_len=0,
              scheduler_metadata=None, max_num_splits=0, causal=True,
              mm_prefix_query_range_tensor=None, rswa_prefix_lens=None,
              num_split_decodes=0, prefill_query_start_loc=None)
    builder = NS(vllm_config=config(), dcp_world_size=1, mach_fp8_fa2=True)
    return builder, common, meta


@pytest.mark.parametrize("lengths,nd", [([1] * 30 + [526, 1492], 30),
                                       ([1] * 60 + [646, 1342], 60),
                                       ([1, 1, 8], 2)])
def test_metadata_rebases_measured_prefills_and_owns_each_build(attention, lengths, nd):
    builder, common, meta = metadata(lengths)
    before = common.query_start_loc.clone()
    attention.build_split_metadata(builder, common, meta)
    assert meta.num_split_decodes == nd
    assert meta.prefill_query_start_loc.tolist() == [0, *torch.tensor(
        lengths[nd:]).cumsum(0).tolist()]
    assert torch.equal(common.query_start_loc, before)
    old = meta.prefill_query_start_loc.clone()
    _, next_common, next_meta = metadata([1, 1, 9, 7])
    attention.build_split_metadata(builder, next_common, next_meta)
    assert next_meta.prefill_query_start_loc.tolist() == [0, 9, 16]
    assert torch.equal(meta.prefill_query_start_loc, old)


@pytest.mark.parametrize("change", ["disabled", "tp", "pp", "dbo", "spec",
                                    "dcp", "cascade", "scheduler", "mm", "rswa",
                                    "unsorted", "pure_decode", "pure_prefill",
                                    "zero_length", "three_prefills", "offsets",
                                    "token_total", "qmax", "starts_identity"])
def test_metadata_unsupported_contract_keeps_single_call(attention, change):
    lengths = {"unsorted": [1, 8, 1], "pure_decode": [1, 1],
               "pure_prefill": [8, 9], "zero_length": [1, 0, 8],
               "three_prefills": [1, 8, 9, 10]}.get(change, [1, 1, 8, 9])
    builder, common, meta = metadata(lengths)
    if change == "disabled": builder.mach_fp8_fa2 = False
    if change == "tp": builder.vllm_config.parallel_config.tensor_parallel_size = 2
    if change == "pp": builder.vllm_config.parallel_config.pipeline_parallel_size = 2
    if change == "dbo": builder.vllm_config.parallel_config.enable_dbo = True
    if change == "spec": builder.vllm_config.speculative_config = object()
    if change == "dcp": builder.dcp_world_size = 2
    if change == "cascade": meta.use_cascade = True
    if change == "scheduler": meta.scheduler_metadata = torch.zeros(1)
    if change == "mm": meta.mm_prefix_query_range_tensor = torch.zeros(1)
    if change == "rswa": meta.rswa_prefix_lens = torch.zeros(1)
    if change == "offsets": common.query_start_loc_cpu[0] = 1
    if change == "token_total": meta.num_actual_tokens += 1
    if change == "qmax": meta.max_query_len += 1
    if change == "starts_identity": meta.query_start_loc = meta.query_start_loc.clone()
    attention.build_split_metadata(builder, common, meta)
    assert meta.num_split_decodes == 0
    assert meta.prefill_query_start_loc is None


class AttentionType(Enum):
    DECODER = 1
    ENCODER = 2
    ENCODER_ONLY = 3


def forward_from_source(attention, recorder):
    relative = "vllm/v1/attention/backends/flash_attn.py"
    override = os.environ.get("VLLM_MACH_FP8_FA2_SOURCE")
    if override:
        path = Path(override)
    else:
        try:
            path = Path(package_metadata.distribution("vllm").locate_file(relative))
        except package_metadata.PackageNotFoundError:
            pytest.skip("Normal-forward checks require the installed block-FP8 runtime")
    manifest = json.loads((REPO / "src/vllm_mach/fp8/data/runtime_sources.json").read_text())
    identity = manifest["files"][relative]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == identity["upstream_sha256"]:
        pytest.skip("Apply vllm-mach-fp8-install before normal-forward checks")
    assert digest == identity["installed_sha256"], "FA2 runtime source identity differs"
    tree = ast.parse(path.read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "FlashAttentionImpl")
    forward = next(node for node in impl.body if isinstance(node, ast.FunctionDef)
                   and node.name == "forward")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                            level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, forward], type_ignores=[]))
    namespace = dict(torch=torch, dataclass=dataclass, AttentionType=AttentionType,
                     canonicalize_singleton_dim_strides=lambda tensor: tensor,
                     is_quantized_kv_cache=lambda dtype: False,
                     _maybe_symmetrize_window=lambda window, causal: window,
                     mixed_fa2_eligible=attention.mixed_fa2_eligible,
                     flash_attn_varlen_func=recorder)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["forward"]


@pytest.mark.parametrize("fallback", [None, "disabled", "decode_only", "prefill_only",
                                     "fa3", "fa4", "capture", "dtype",
                                     "window", "softcap", "alibi", "sinks",
                                     "num_splits", "head_size", "page_size",
                                     "causal", "scheduler"])
def test_normal_forward_preserves_request_slices_cache_and_output(attention, monkeypatch,
                                                                 fallback):
    lengths = {"decode_only": [1, 1], "prefill_only": [5, 7]}.get(
        fallback, [1, 1, 5, 7])
    builder, common, meta = metadata(lengths)
    if fallback == "disabled": builder.mach_fp8_fa2 = False
    attention.build_split_metadata(builder, common, meta)
    actual, padded = meta.num_actual_tokens, meta.num_actual_tokens + 3
    dtype = torch.float16 if fallback == "dtype" else torch.bfloat16
    hd = 128 if fallback == "head_size" else 256
    page = 512 if fallback == "page_size" else 528
    q = torch.arange(padded, dtype=dtype)[:, None, None].expand(padded, 16, hd).contiguous()
    output = torch.full_like(q, -9)
    cache = torch.zeros((2, 4, page, 2 * hd), dtype=dtype)
    impl = NS(vllm_flash_attn_version={"fa3": 3, "fa4": 4}.get(fallback, 2),
              attn_type=AttentionType.DECODER, head_size=hd, num_heads=16,
              num_kv_heads=4, supports_quant_query_input=False,
              kv_cache_dtype="auto", dcp_world_size=1, scale=0.0625,
              sliding_window=(128, 0) if fallback == "window" else None,
              logits_soft_cap=1 if fallback == "softcap" else 0,
              alibi_slopes=torch.ones(16) if fallback == "alibi" else None,
              sinks=torch.ones(16) if fallback == "sinks" else None,
              fa4_hd256=False)
    if fallback == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    if fallback == "num_splits": meta.max_num_splits = 2
    if fallback == "causal": meta.causal = False
    if fallback == "scheduler": meta.scheduler_metadata = torch.zeros(1)
    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        kwargs["out"].copy_(kwargs["q"])
        return kwargs["out"]

    forward = forward_from_source(attention, recorder)
    layer = NS(_q_scale=torch.tensor(1.), _k_scale=torch.tensor(2.),
               _v_scale=torch.tensor(3.))
    result = forward(impl, layer, q, q[:, :4], q[:, :4], cache, meta, output)
    assert result is output
    assert torch.equal(output[:actual], q[:actual])
    assert (output[actual:] == -9).all()
    assert len(calls) == (2 if fallback is None else 1)
    if fallback is not None:
        assert calls[0]["cu_seqlens_q"] is meta.query_start_loc
        assert calls[0]["num_splits"] == meta.max_num_splits
        return
    decode, prefill = calls
    assert decode["cu_seqlens_q"].tolist() == [0, 1, 2]
    assert prefill["cu_seqlens_q"].tolist() == [0, 5, 12]
    assert [call["max_seqlen_q"] for call in calls] == [1, 7]
    for call, reqs, tokens in ((decode, slice(0, 2), slice(0, 2)),
                               (prefill, slice(2, None), slice(2, actual))):
        assert torch.equal(call["seqused_k"], meta.seq_lens[reqs])
        assert torch.equal(call["block_table"], meta.block_table[reqs])
        assert call["out"].data_ptr() == output[tokens].data_ptr()
        assert call["q"].data_ptr() == q[tokens].data_ptr()
        assert call["k"].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert call["v"].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert call["k_descale"].shape == (2, 4)
        assert (call["k_descale"] == 2).all()
        assert (call["v_descale"] == 3).all()
        assert call["q_descale"] is None
        assert call["softmax_scale"] == 0.0625
        assert call["causal"] is True
        assert call["num_splits"] == 1
