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
def test_layer_geometry_and_native_semantics_guard(change):
    from vllm_mach.mxfp6.gdn_decode import _eligible_layer

    layer = type("QwenGatedDeltaNetAttention", (), {})()
    vars(layer).update(
        tp_size=2,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        enable_fused_gdn_decode=True,
        enable_packed_recurrent_decode=True,
        activation="silu",
        in_proj_ba=SimpleNamespace(weight=torch.empty(48, 5120, dtype=torch.bfloat16)),
        conv1d=SimpleNamespace(weight=torch.empty(5120, 1, 4, dtype=torch.bfloat16)),
        A_log=torch.empty(24),
        dt_bias=torch.empty(24, dtype=torch.bfloat16),
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
