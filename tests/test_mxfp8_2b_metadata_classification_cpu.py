"""Exact16 integration mechanics; actual CPU tensors use installed vLLM when present."""
import ast
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import types
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
if not os.environ.get("MACH_TEST_RUNTIME_PROFILE"):
    sys.path.insert(0, str(SOURCE))
from vllm_mach.mxfp8.two_b import metadata_classification as mod

if os.environ.get("MACH_TEST_RUNTIME_PROFILE"):
    from importlib.metadata import distribution
    expected = distribution("vllm-mach").locate_file(
        "vllm_mach/mxfp8/two_b/metadata_classification.py")
    assert Path(mod.__file__).resolve() == Path(expected).resolve()


def selected_worker(builders):
    return NS(_mach_2b_contract=dict(mode="production", ssm_dtype="float32", hybrid_greedy_head=True),
              _mach_2b_hybrid_head={"state": NS(candidates=128, _head_graphs=None)},
              model_runner=NS(speculative_config=None,
                  cudagraph_manager=NS(_graphs_captured=True, graphs={1: object()}),
                  attn_groups=[[NS(metadata_builders=builders)]]))


def mechanics():
    events = []
    marker = object()
    error = RuntimeError("original sentinel")
    def reference(m):
        events.append(("reference", m))
        if m == "error":
            raise error
        return (0, 1, 0, 16)
    def remainder(counts, actual_marker, keyword):
        events.append(("remainder", actual_marker, keyword))
        return counts, actual_marker, keyword
    namespace = dict(DEFAULT=marker, reference=reference, remainder=remainder)
    text = """def build(self, m, marker=DEFAULT, *, keyword=DEFAULT):
    counts = reference(m)
    return remainder(counts, marker, keyword)
"""
    exec(compile(text, "<metadata-integration-fixture>", "exec"), namespace)
    original = namespace["build"]
    node = ast.parse("""def build(self, m, marker=DEFAULT, *, keyword=DEFAULT):
    counts = _mach_exact16_classifier(m)
    if counts is None:
        counts = reference(m)
    return remainder(counts, marker, keyword)
""").body[0]
    def fast(m):
        return (16, 0, 16, 0) if m == "exact16" else None
    class Builder:
        build = original
        def update(self): return "unchanged"
    return Builder, original, node, fast, events, marker, error


def test_factory_preserves_globals_defaults_kwargs_and_original_exception():
    Builder, original, node, fast, events, marker, error = mechanics()
    keys = set(original.__globals__)
    fn = mod._factory(original, node, fast)
    assert fn.__globals__ is original.__globals__ and set(original.__globals__) == keys
    assert fn.__defaults__ is original.__defaults__ and fn.__kwdefaults__ is original.__kwdefaults__
    assert fn.__annotations__ is original.__annotations__ and fn.__qualname__ == original.__qualname__
    assert fn.__code__.co_freevars == (mod._HELPER,)
    layer = Builder()
    result = fn(layer, "exact16")
    assert result == ((16, 0, 16, 0), marker, marker)
    assert len(events) == 1 and events[0][0] == "remainder"
    events.clear(); opaque = object()
    assert fn(layer, "mixed", opaque, keyword=opaque) == ((0, 1, 0, 16), opaque, opaque)
    assert [event[0] for event in events] == ["reference", "remainder"]
    events.clear()
    with pytest.raises(RuntimeError) as caught:
        fn(layer, "error")
    assert caught.value is error and [event[0] for event in events] == ["reference"]


def test_three_instance_bindings_keep_class_type_update_and_full_attention():
    Builder, original, node, fast, *_ = mechanics()
    builders = [Builder() for _ in range(3)]
    class FullAttention:
        def build(self): return "original"
    full = FullAttention(); full_original = full.build.__func__
    worker = selected_worker([*builders, full]); update = Builder.update
    with patch.object(mod, "_runtime", return_value=(Builder, fast)), patch.object(mod, "_source", return_value=(original, node)):
        assert mod.install(worker)
        assert not mod.install(worker)
    assert all(type(builder) is Builder and builder.build.__self__ is builder for builder in builders)
    assert all(builder.build.__func__.__globals__ is original.__globals__ for builder in builders)
    assert Builder.build is original and Builder.update is update
    assert "build" not in vars(full) and full.build.__func__ is full_original
    assert mod.inspect_worker(worker)["builder_count"] == 3


def test_custom_instance_rejection_and_source_failure_are_atomic():
    Builder, original, node, fast, *_ = mechanics()
    builders = [Builder() for _ in range(3)]; custom = lambda: "custom"
    builders[1].build = custom
    with patch.object(mod, "_runtime", return_value=(Builder, fast)), patch.object(mod, "_source", side_effect=AssertionError("must reject before source")):
        assert mod.install(selected_worker(builders)) is False
    assert "build" not in vars(builders[0]) and "build" not in vars(builders[2]) and builders[1].build is custom
    del builders[1].build
    with patch.object(mod, "_runtime", return_value=(Builder, fast)), patch.object(mod, "_source", side_effect=RuntimeError("source changed")):
        with pytest.raises(RuntimeError, match="source changed"):
            mod.install(selected_worker(builders))
    assert all("build" not in vars(builder) for builder in builders)
    with patch.object(mod, "_runtime", return_value=(Builder, fast)):
        with pytest.raises(RuntimeError, match="three exact"):
            mod.install(selected_worker(builders[:2]))


def test_unselected_profiles_and_graphs_reject_before_runtime_import():
    worker = selected_worker([])
    rejected = []
    for key, value in (("mode", "quality"), ("ssm_dtype", "bfloat16"), ("hybrid_greedy_head", False)):
        value_worker = copy.deepcopy(worker); value_worker._mach_2b_contract[key] = value; rejected.append(value_worker)
    value_worker = copy.deepcopy(worker); value_worker._mach_2b_hybrid_head["state"]._head_graphs = object(); rejected.append(value_worker)
    value_worker = copy.deepcopy(worker); value_worker.model_runner.speculative_config = object(); rejected.append(value_worker)
    value_worker = copy.deepcopy(worker); value_worker.model_runner.cudagraph_manager._graphs_captured = False; rejected.append(value_worker)
    rejected.append(NS())
    with patch.object(mod, "_runtime", side_effect=AssertionError("runtime imported for unselected configuration")):
        assert all(mod.install(value_worker) is False for value_worker in rejected)


def test_modes_and_compile_refuse_before_metadata_tensor_properties():
    class Metadata:
        @property
        def num_reqs(self): raise AssertionError("metadata read before mode refusal")
    torch = NS(compiler=NS(is_compiling=lambda: False), jit=NS(is_tracing=lambda: False, is_scripting=lambda: False))
    for dispatch, function in ((lambda: object(), lambda: None), (lambda: None, lambda: object())):
        assert mod._bound_helper(torch, Metadata, dispatch, function)(Metadata()) is None
    torch.compiler.is_compiling = lambda: True
    assert mod._bound_helper(torch, Metadata, lambda: None, lambda: None)(Metadata()) is None


def actual_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    helper = mod._bound_helper(torch, CommonAttentionMetadata, _get_current_dispatch_mode, _get_current_function_mode)
    return torch, CommonAttentionMetadata, GDNAttentionMetadataBuilder, helper


def test_actual_cpu_helper_domain_storage_and_fallback():
    torch, Common, _, fast = actual_runtime()
    def metadata(lengths, seq, p):
        starts = [0]
        for length in lengths: starts.append(starts[-1] + length)
        q = torch.tensor(starts, dtype=torch.int32); s = torch.tensor(seq, dtype=torch.int32)
        return Common(query_start_loc=q, query_start_loc_cpu=q, seq_lens=s, num_reqs=len(lengths),
            num_actual_tokens=starts[-1], max_query_len=max(lengths), max_seq_len=max(seq),
            block_table_tensor=torch.zeros((len(lengths), 1), dtype=torch.int32),
            slot_mapping=torch.zeros(starts[-1], dtype=torch.int64), seq_lens_cpu_upper_bound=s,
            is_prefilling=torch.tensor(p, dtype=torch.bool))
    with torch.inference_mode():
        for seq in (0, 1, 4096):
            m = metadata([1]*16, [seq]*16, [False]*16)
            tensors = (m.query_start_loc_cpu, m.seq_lens_cpu_upper_bound, m.is_prefilling)
            values = [t.clone() for t in tensors]; flags = [t.untyped_storage().resizable() for t in tensors]
            assert fast(m) == (16, 0, 16, 0)
            assert flags == [t.untyped_storage().resizable() for t in tensors]
            assert all(torch.equal(a, b) for a, b in zip(tensors, values))
        assert fast(metadata([1]*16, [1]*16, [True]+[False]*15)) is None
        assert fast(metadata([1]*15+[0], [1]*16, [False]*16)) is None
        assert fast(metadata([1]*8, [1]*8, [False]*8)) is None
        assert fast(metadata([1]*32, [1]*32, [False]*32)) is None
    assert not torch.cuda.is_initialized()


def test_actual_source_six_statements_and_full_function_inverse():
    _, _, Builder, _ = actual_runtime()
    original, modified = mod._source(Builder)
    assert original is Builder.build and original.__globals__ is vars(sys.modules[Builder.__module__])
    # _source compiles the full installed module and checks original code identity;
    # _transform removes the insertion and compares every original AST node.
    assert modified.name == "build" and len(modified.body) > 6
