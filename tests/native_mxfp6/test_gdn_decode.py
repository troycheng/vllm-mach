"""Native GDN routing must preserve unsupported and mixed-batch fallbacks."""

from types import SimpleNamespace

import pytest
import torch

from vllm_mach.mxfp6.gdn_decode import select_path
from vllm_mach.mxfp6.serve import build_command


def metadata(rows, **changes):
    return SimpleNamespace(
        **dict(
            dict(
                spec_sequence_masks=None,
                num_spec_decodes=0,
                num_prefills=0,
                num_decodes=rows,
                num_actual_tokens=rows,
                non_spec_state_indices_tensor=object(),
            ),
            **changes,
        )
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 24, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_paths_are_disjoint_and_dtype_guarded(rows, dtype):
    actual = select_path(rows, dtype, metadata(rows), persistent=True, overlap=True)
    expected = (
        "persistent"
        if rows in (1, 2, 4, 8) and dtype in (torch.float32, torch.float16)
        else "overlap"
        if rows in (16, 24, 32) and dtype in (torch.float32, torch.float16)
        else None
    )
    assert actual == expected


@pytest.mark.parametrize(
    "changes",
    [
        dict(num_prefills=1),
        dict(num_spec_decodes=1),
        dict(spec_sequence_masks=object()),
        dict(num_decodes=0),
        dict(num_actual_tokens=7),
        dict(non_spec_state_indices_tensor=None),
    ],
)
@pytest.mark.parametrize("rows", [4, 32])
def test_unsupported_metadata_falls_back(rows, changes):
    assert (
        select_path(
            rows,
            torch.float32,
            metadata(rows, **changes),
            persistent=True,
            overlap=True,
        )
        is None
    )


def test_disabled_and_missing_metadata():
    assert select_path(4, torch.float32, None, persistent=True, overlap=True) is None
    assert (
        select_path(4, torch.float32, metadata(4), persistent=False, overlap=False)
        is None
    )


def test_compiled_gdn_out_schema_and_caller_ownership(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm.config import CompilationMode
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm_mach.mxfp6 import gdn_decode

    owner = torch.nn.Module()
    config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE, splitting_ops=[])
    owner.vllm_config = SimpleNamespace(compilation_config=config)
    assert gdn_decode._prepare_compiled_dispatch(owner)
    assert gdn_decode._prepare_compiled_dispatch(owner)
    assert config.splitting_ops == ["vllm::mach_gdn_forward_out"]
    op = torch.ops.vllm.mach_gdn_forward_out.default
    schema = op._schema
    assert schema.arguments[0].alias_info is None
    assert schema.arguments[1].name == "output"
    assert schema.arguments[1].alias_info.is_write and not schema.returns

    layer = SimpleNamespace(_mach_gdn_original=None, _mach_gdn_persistent=True,
                            _mach_gdn_aux=None)
    context = ForwardContext({"test": layer}, {}, {})
    monkeypatch.setattr(gdn_decode, "_forward", lambda *args: args[-1].clone() * 2)
    x = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    initial = x.clone()
    a, b = torch.empty_like(x), torch.empty_like(x)
    with override_forward_context(context):
        gdn_decode._compiled_forward_out(x, a, "test")
        saved = a.clone()
        gdn_decode._compiled_forward_out(x + 1, b, "test")
    assert torch.equal(x, initial) and torch.equal(a, saved)
    assert torch.equal(a, x * 2) and torch.equal(b, (x + 1) * 2)

    # Check the compiled frontend's dataflow, not an incidental alias label.
    graph = torch.fx.symbolic_trace(lambda x: gdn_decode._compiled_dispatch("test", x))
    allocation = next(n for n in graph.graph.nodes if n.target is torch.empty_like)
    call = next(n for n in graph.graph.nodes
                if n.target is torch.ops.vllm.mach_gdn_forward_out)
    returned = next(n for n in graph.graph.nodes if n.op == "output")
    assert call.args[1] is allocation and returned.args[0] is allocation
    with FakeTensorMode():
        fake = torch.empty(4, 4, dtype=torch.bfloat16)
        assert op(fake, torch.empty_like(fake), "test") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@torch.inference_mode()
def test_compiled_gdn_piecewise_changed_inputs(monkeypatch, default_vllm_config):
    """An eager split op must write the buffer a downstream graph captured."""
    from vllm.compilation import monitor
    from vllm.compilation.cuda_graph import CUDAGraphWrapper, CUDAGraphOptions
    from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config
    from vllm.forward_context import BatchDescriptor, ForwardContext, override_forward_context
    from vllm_mach.mxfp6 import gdn_decode

    # The fixture establishes a context; set_current_vllm_config yields None.
    config = get_current_vllm_config()
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    config.compilation_config.splitting_ops = []
    owner = torch.nn.Module()
    owner.vllm_config = config
    assert gdn_decode._prepare_compiled_dispatch(owner)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", True)
    layer = SimpleNamespace(_mach_gdn_original=None, _mach_gdn_persistent=True,
                            _mach_gdn_aux=None)
    context = ForwardContext({"test": layer}, {}, {},
                             cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                             batch_descriptor=BatchDescriptor(num_tokens=56))
    fresh_results = []
    factor = [2]

    def fresh_forward(*args):
        result = args[-1] * factor[0]
        fresh_results.append(result)  # Prevent allocator reuse hiding the bug.
        return result

    monkeypatch.setattr(gdn_decode, "_forward", fresh_forward)
    options = CUDAGraphOptions(weak_ref_output=False)
    before = CUDAGraphWrapper(lambda x: (x.clone(), torch.empty_like(x)), config,
                             CUDAGraphMode.PIECEWISE, options)
    after = CUDAGraphWrapper(lambda output: output + 1, config,
                            CUDAGraphMode.PIECEWISE, options)
    before.is_debugging_mode = after.is_debugging_mode = True
    x = torch.zeros(56, 16, device="cuda", dtype=torch.bfloat16)
    a = torch.arange(x.numel(), device="cuda").reshape_as(x).to(x.dtype) / 128
    b = -a - 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), override_forward_context(context):
        # Warm ordinary kernels before real vLLM piecewise captures.
        warm_x, warm_out = before.runnable(x)
        torch.ops.vllm.mach_gdn_forward_out(warm_x, warm_out, "test")
        after.runnable(warm_out)
        strong_storage = before(x)
        static_address = strong_storage[1].data_ptr()
        # A wrapper's first call only captures; the captured kernels have not
        # populated its output yet. Capture both pieces before scoring replays.
        graph_x, output = before(x)
        torch.ops.vllm.mach_gdn_forward_out(graph_x, output, "test")
        strong_result = after(output)
        graph_x, output = before(x)
        torch.ops.vllm.mach_gdn_forward_out(graph_x, output, "test")
        assert torch.equal(after(output), x * 2 + 1)
        for value, multiplier in [(a, 2), (b, 1), (a, 2)]:
            x.copy_(value)
            factor[0] = multiplier
            graph_x, output = before(x)
            assert output.data_ptr() == static_address
            torch.ops.vllm.mach_gdn_forward_out(graph_x, output, "test")
            result = after(output)
            assert torch.equal(result, value * multiplier + 1)
    torch.cuda.current_stream().wait_stream(stream)


def test_launcher_defaults_state_fallback_and_resets_flags(monkeypatch):
    monkeypatch.setenv("VLLM_MACH_GDN_PERSISTENT", "1")
    monkeypatch.setenv("VLLM_MACH_GDN_BA_OVERLAP", "1")
    args = SimpleNamespace(
        model="model",
        fp16_ssm=False,
        lossless_prefill=False,
        owner_prefill=False,
        nvfp4_lm_head=False,
        verify_prefill=False,
        gdn_persistent=False,
        gdn_ba_overlap=False,
    )
    _, env = build_command(args, [])
    assert env["VLLM_MACH_GDN_PERSISTENT"] == env["VLLM_MACH_GDN_BA_OVERLAP"] == "0"
    args.gdn_persistent = True
    args.fp16_ssm = True
    _, env = build_command(args, [])
    assert env["VLLM_MACH_GDN_PERSISTENT"] == "1"
    del args.gdn_persistent, args.gdn_ba_overlap
    args.fp16_ssm = False
    _, env = build_command(args, [])
    assert env["VLLM_MACH_GDN_PERSISTENT"] == env["VLLM_MACH_GDN_BA_OVERLAP"] == "1"


@pytest.mark.parametrize(
    "change",
    [
        {"tp_size": 1},
        {"num_k_heads": 32},
        {"num_v_heads": 64},
        {"head_k_dim": 64},
        {"gqa_interleaved_layout": True},
        {"disable_tp_for_ba_proj": True},
        {"enable_fused_gdn_decode": False},
        {"enable_packed_recurrent_decode": False},
        {"activation": "relu"},
    ],
)
@pytest.mark.parametrize("hidden, heads", [(5120, 24), (2048, 16)])
def test_layer_geometry_and_native_semantics_guard(change, hidden, heads):
    from vllm_mach.mxfp6.gdn_decode import _eligible_layer

    layer = type("QwenGatedDeltaNetAttention", (), {})()
    vars(layer).update(
        tp_size=2,
        num_k_heads=16,
        num_v_heads=2 * heads,
        head_k_dim=128,
        head_v_dim=128,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        enable_fused_gdn_decode=True,
        enable_packed_recurrent_decode=True,
        activation="silu",
        in_proj_ba=SimpleNamespace(
            weight=torch.empty(2 * heads, hidden, dtype=torch.bfloat16)
        ),
        conv1d=SimpleNamespace(
            weight=torch.empty((16 + heads) * 128, 1, 4, dtype=torch.bfloat16)
        ),
        A_log=torch.empty(heads),
        dt_bias=torch.empty(heads, dtype=torch.bfloat16),
        norm=SimpleNamespace(weight=torch.empty(128)),
    )
    assert _eligible_layer(layer)
    vars(layer).update(change)
    assert not _eligible_layer(layer)


def test_persistent_state_dtype_is_part_of_jit_and_scratch_key():
    from vllm_mach.mxfp6.gdn import persistent

    x = torch.empty(4, 5120)
    ba = torch.empty(5120, 48)
    qkv = torch.empty(4, 5120)
    conv = torch.empty(8, 5120, 3)
    weight = torch.empty(5120, 4)
    al = torch.empty(24)
    keys = []
    for dtype in (torch.float32, torch.float16):
        state = torch.empty(8, 24, 128, 128, dtype=dtype)
        key = persistent._geometry_from_tensors(x, ba, qkv, weight, conv, al, state)
        keys.append(key)
    assert keys[0][:-1] == keys[1][:-1]
    assert keys[0][-1] == 32 and keys[1][-1] == 16
    assert persistent._scratch_key(keys[0], x, conv) != persistent._scratch_key(
        keys[1], x, conv
    )
    with pytest.raises(ValueError, match="FP16 or FP32"):
        persistent._geometry_from_tensors(
            x, ba, qkv, weight, conv, al, state.bfloat16()
        )


@pytest.mark.parametrize(
    "tp, hidden, total_heads, expected",
    [
        (2, 5120, 48, (5120, 24, 5120)),
        (2, 2048, 32, (2048, 16, 4096)),
        (1, 2560, 32, (2560, 32, 8192)),
    ],
)
def test_geometry_keeps_hidden_and_qkv_widths_distinct(
    tp, hidden, total_heads, expected
):
    from vllm_mach.mxfp6.gdn_decode import _layer_geometry

    layer = SimpleNamespace(
        tp_size=tp,
        in_proj_ba=SimpleNamespace(weight=torch.empty(total_heads, hidden)),
        num_v_heads=total_heads,
        num_k_heads=16,
    )
    assert _layer_geometry(layer) == expected


def test_graph_readiness_does_not_reuse_dense_geometry(monkeypatch):
    from vllm_mach.mxfp6.gdn import persistent

    x = torch.empty(4, 2048)
    conv = torch.empty(6, 4096, 3)
    signature = dict(
        hidden=2048,
        n_ba=32,
        qkv_dim=4096,
        h_q=8,
        hv=16,
        d=128,
        conv_width=4,
        conv_state_len=3,
        state_bits=32,
    )
    key = persistent.geometry_key(signature)
    dense_key = (5120, 48, 5120, 8, 24, 128, 4, 3, 32)
    modules = {dense_key: object()}
    scratch = {persistent._scratch_key(dense_key, x, conv): object()}
    monkeypatch.setattr(persistent, "_modules", modules)
    monkeypatch.setattr(persistent, "_scratch_cache", scratch)
    monkeypatch.setattr(persistent, "_barrier_cache", {"cpu": object()})
    assert not persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)
    modules[key] = object()
    assert not persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)
    scratch[persistent._scratch_key(key, x, conv)] = object()
    assert persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)


@pytest.mark.parametrize("tp, hidden, heads", [(2, 5120, 24), (2, 2048, 16), (1, 2560, 32)])
@pytest.mark.parametrize("rows", [4, 16, 24, 32])
@torch.inference_mode()
def test_gpu_adapter_matches_native_decode(monkeypatch, tp, hidden, heads, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    import vllm.forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    from vllm_mach.mxfp6.gdn_decode import _forward

    torch.manual_seed(20260917)
    qkv_dim = (2 * (16 // tp) + heads) * 128

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    x = rand(rows, hidden)
    weight = rand(2 * heads, hidden)
    mixed = rand(rows, qkv_dim + heads * 128)
    cw, cb = rand(qkv_dim, 4), rand(qkv_dim)
    conv = rand(rows + 1, 3, qkv_dim).transpose(1, 2)
    if is_conv_state_dim_first():
        conv = conv.contiguous()
    state = rand(rows + 1, heads, 128, 128).float()
    reference_conv, reference_state = conv.clone(), state.clone()
    indices = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    md = metadata(rows, non_spec_state_indices_tensor=indices)
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test": md}),
    )

    class BA:
        def __init__(self):
            self.weight = weight

        def __call__(self, value):
            return torch.nn.functional.linear(value, self.weight), None

    # A deterministic gated output makes QKV/Z splitting observable without
    # testing vLLM's unrelated RMSNorm implementation again.
    def norm(core, z, out):
        out.copy_(core * torch.nn.functional.silu(z))

    layer = SimpleNamespace(
        tp_size=tp,
        num_k_heads=16,
        num_v_heads=heads * tp,
        in_proj_ba=BA(),
        in_proj_qkvz=lambda _: (mixed, None),
        split_ba=lambda ba: ba.chunk(2, -1),
        conv1d=SimpleNamespace(weight=cw, bias=cb),
        _mach_gdn_ba=weight.T.contiguous(),
        _mach_gdn_bias=cb,
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=rand(heads),
        activation="silu",
        prefix="test",
        kv_cache=(conv if is_conv_state_dim_first() else conv.transpose(-1, -2), state),
        _rms_norm_gated_cuda=norm,
        out_proj=lambda value: (value, None),
    )
    aux = torch.cuda.Stream() if rows >= 16 else None

    def unexpected_fallback(_):
        pytest.fail("supported geometry unexpectedly fell back")

    # Native convolution overwrites the packed QKV view in place.
    reference_mixed = mixed.clone()
    actual = _forward(layer, unexpected_fallback, True, aux, x)
    ba, _ = layer.in_proj_ba(x)
    b, a = ba.chunk(2, -1)
    qkv, z = reference_mixed.split([qkv_dim, heads * 128], -1)
    cq = causal_conv1d_update(
        qkv,
        reference_conv,
        cw,
        cb,
        "silu",
        conv_state_indices=indices,
        validate_data=False,
    )
    expected = torch.empty(rows, 1, heads, 128, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=cq,
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        scale=128**-0.5,
        initial_state=reference_state,
        out=expected,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    norm(expected, z.reshape_as(expected), expected)
    torch.cuda.synchronize()
    assert torch.equal(conv, reference_conv)
    for result, reference in ((actual, expected.flatten(1)), (state, reference_state)):
        relative = (
            result.float() - reference.float()
        ).norm() / reference.float().norm()
        assert relative < 0.02
        if aux is not None:
            assert torch.equal(result, reference)


@pytest.mark.parametrize("rows", [1, 2, 8])
def test_tp1_small_batch_adapter(monkeypatch, rows):
    test_gpu_adapter_matches_native_decode(monkeypatch, 1, 2560, 32, rows)
