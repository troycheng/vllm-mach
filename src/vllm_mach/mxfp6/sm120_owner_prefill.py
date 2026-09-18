# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-mach 8c5021a (Apache-2.0).

"""Opt-in owner residual with padded ragged transport and original MLP/BA.

Only the owner half of the full-shaped residual tensor is valid between layers.
The model-wide state makes that contract explicit. It never reaches old norms.
"""

import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from vllm import envs
from vllm.platforms import current_platform

H = 5120


def replica_layers():
    raw = json.loads(envs.VLLM_SM120_OWNER_MLP_LAYERS)
    if (
        not isinstance(raw, list)
        or any(type(i) is not int or not 0 <= i < 64 for i in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(
            "VLLM_SM120_OWNER_MLP_LAYERS must be distinct layer indices in [0, 64)"
        )
    return tuple(raw)


MLP_LAYERS = replica_layers()


def row_partition(rows, rank):
    if not 512 <= rows <= 4096 or rank not in (0, 1):
        raise ValueError("Owner prefill requires M512..4096 and TP rank 0 or 1")
    packet_rows = ((rows + 255) // 256) * 128
    counts = (packet_rows, rows - packet_rows)
    own = slice(rank * packet_rows, rank * packet_rows + counts[rank])
    other = slice((1 - rank) * packet_rows, (1 - rank) * packet_rows + counts[1 - rank])
    return packet_rows, counts, own, other


def packed_weight(module, *, merged=True):
    mxfp6 = import_module("mxfp6")
    from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel

    assert isinstance(module.scheme.ocp_mx_linear, Mxfp6Sm120LinearKernel)
    n, packed_k = module.weight.shape
    return mxfp6.PackedMXFP6Tensor(
        module.weight, module.weight_scale, n, packed_k * 4 // 3
    )


def apply_mxfp8_weight(activation, weight):
    mxfp6 = import_module("mxfp6")
    return mxfp6.gemm_w6a8(activation, weight, out_dtype=torch.bfloat16)


def load_native():
    from .sm120_owner_prefill_native import load

    load()


def gathered_rows(x):
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x.contiguous(), dim=0)


def padded_rows(state, x):
    """All-gather unequal owner rows through equal P-row transport packets."""
    assert x.shape[0] == state.local_rows
    if state.m == 4096:
        return gathered_rows(x)
    padded = torch.zeros((state.p, *x.shape[1:]), dtype=x.dtype, device=x.device)
    padded[: state.local_rows].copy_(x)
    return gathered_rows(padded)[: state.m]


def prepare_mlp_replicas(state):
    mxfp6 = import_module("mxfp6")
    load_native()
    state.mlp_weights = {}
    state.local_mlp_partials = None
    total = 0
    for layer_id in MLP_LAYERS:
        layer = state.model.layers[layer_id]
        branches: list[dict[str, Any]] = [{}, {}]
        for key, projection in (("gu", "gate_up_proj"), ("down", "down_proj")):
            module = getattr(layer.mlp, projection)
            assert module.bias is None
            if key == "gu":
                original = packed_weight(module)
            else:
                assert module.input_is_parallel and not module.reduce_results
                original = packed_weight(module, merged=False)
            assert original is not None
            assert (original.rows, original.k) == (
                (17408, H) if key == "gu" else (H, 8704)
            )
            buffers = {}
            for field in ("values", "scales"):
                source = getattr(original, field)
                assert source.dtype == torch.uint8 and source.is_contiguous()
                both = gathered_rows(source.reshape(-1))
                partner = (
                    both.reshape(2, -1)[1 - state.rank].clone().reshape(source.shape)
                )
                del both
                buffers[field] = partner
                total += partner.numel()
            branches[state.rank][key] = original
            branches[1 - state.rank][key] = mxfp6.PackedMXFP6Tensor(
                buffers["values"], buffers["scales"], original.rows, original.k
            )
        state.mlp_weights[layer_id] = branches
    assert total == len(MLP_LAYERS) * 104448000
    state.replica_bytes = total
    print(
        json.dumps(
            {
                "owner_local_mlp_initialized": True,
                "rank": state.rank,
                "layers": list(MLP_LAYERS),
                "partner_bytes": total,
            }
        ),
        flush=True,
    )


def local_mlp(state, layer, normalized, label):
    mxfp6 = import_module("mxfp6")
    autotune = import_module("mxfp6.autotune")
    W6A8Config, _run_config = autotune.W6A8Config, autotune._run_config
    x = normalized.prepared
    assert x.rows == state.local_rows and x.k == H
    partials = []
    for weights in state.mlp_weights[state.indices[id(layer)]]:
        gu, down = weights["gu"], weights["down"]
        y = _run_config(
            W6A8Config(12, 1, 2),
            x.values,
            gu.values,
            x.scales,
            gu.scales,
            state.local_rows,
            gu.rows,
            H,
            torch.bfloat16,
        )
        values = torch.empty(
            (state.local_rows, 8704), dtype=torch.uint8, device=y.device
        )
        scales = torch.empty(
            ((state.local_rows + 127) // 128 * 128 * 272,),
            dtype=torch.uint8,
            device=y.device,
        )
        mxfp6.silu_and_mul_mxfp8_packed_out(values, scales, y)
        partial = _run_config(
            W6A8Config(14, 1, 1),
            values,
            down.values,
            scales,
            down.scales,
            state.local_rows,
            down.rows,
            8704,
            torch.bfloat16,
        )
        partials.append(partial)
    reference = None
    if state.verify:
        reference = layer.mlp(normalized.oracle_norm)
        both = gathered_rows(reference).reshape(2, state.m, H)
        for rank in (0, 1):
            check(
                state, label + f"/mlp_tp{rank}", partials[rank], both[rank, state.own]
            )
    assert state.local_mlp_partials is None
    # Carry the two independently rounded TP outputs to the next norm. No
    # collective may consume this placeholder: norm_step takes this state.
    state.local_mlp_partials = SimpleNamespace(parts=partials, oracle_partial=reference)
    return torch.empty_like(normalized.norm)


def real_prefill(model):
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    first = next(
        layer.linear_attn
        for layer in model.layers
        if layer.layer_type == "linear_attention"
    )
    # MRV2's autotune dummy runs can have attention metadata but no GDN entry.
    item = metadata.get(first.prefix)
    return item is not None and getattr(item, "num_prefills", 0) > 0


def begin(model, hidden):
    if not envs.VLLM_SM120_OWNER_PREFILL or hidden.ndim != 2 or hidden.shape[1] != H:
        return None
    m = int(hidden.shape[0])
    if not 512 <= m <= 4096:
        return None
    # The existing lossless kernels are faster for these small exact shapes.
    if m in (1024, 1052) and envs.VLLM_SM120_LOSSLESS_PREFILL:
        return None
    # No mid-model fallback: establish one complete model-wide contract first.
    if (
        hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or model.aux_hidden_state_layers
    ):
        return None
    if hidden.device.type != "cuda" or not current_platform.is_device_capability(120):
        raise RuntimeError("Owner prefill requires an SM120 CUDA device")
    if torch.cuda.is_current_stream_capturing():
        return None
    state = getattr(model, "_owner_prefill_state", None)
    if state is None:
        import flashinfer.comm
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        assert get_tensor_model_parallel_world_size() == 2 and len(model.layers) == 64
        assert model.start_layer == 0 and model.end_layer == 64
        for layer in model.layers:
            assert layer.use_fused_ar_gemma_norm and not layer.layer_scale
            norms = (layer.input_layernorm, layer.post_attention_layernorm)
            assert all(
                n.weight.dtype == torch.bfloat16 and n.variance_epsilon == 1e-6
                for n in norms
            )
            projection = (
                layer.linear_attn.in_proj_qkvz
                if layer.layer_type == "linear_attention"
                else layer.self_attn.qkv_proj
            )
            assert projection.bias is None and packed_weight(projection).k == H
            assert packed_weight(layer.mlp.gate_up_proj).k == H
            assert packed_weight(layer.mlp.down_proj, merged=False).k == 8704
            assert not layer.mlp.down_proj.reduce_results
        assert (
            model.norm.weight.dtype == torch.bfloat16
            and model.norm.variance_epsilon == 1e-6
        )
        load_native()
        rank = get_tensor_model_parallel_rank()
        # Dedicated workspace: never resize or replace a workspace referenced
        # by the existing M32 CUDA Graphs.
        workspace = flashinfer.comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=2,
            rank=rank,
            max_token_num=4128,
            hidden_dim=H,
            dtype=torch.bfloat16,
            group=get_tp_group().cpu_group,
        )
        assert (
            workspace.backend == "trtllm"
            and int(workspace.metadata["buffer_size"]) >= 84213760
        )
        state = SimpleNamespace(
            rank=rank,
            workspace=workspace,
            ba_weights={},
            indices={},
            verified_forwards=0,
            checks=[],
            active_checks=[],
            verify=False,
            total_forwards=0,
            model=model,
            mlp_weights={},
            local_mlp_partials=None,
            replica_bytes=0,
            mlp_replicas_prepared=False,
        )
        for i, layer in enumerate(model.layers):
            state.indices[id(layer)] = i
            if layer.layer_type == "linear_attention":
                assert layer.linear_attn.enable_fused_gdn_decode
                weight = layer.linear_attn.in_proj_ba.weight.detach()
                assert (
                    weight.dtype == torch.bfloat16
                    and tuple(weight.shape) == (48, H)
                    and weight.is_contiguous()
                )
                # Replicate the actual loaded BF16 parameters, not a guessed
                # checkpoint dtype or an independently re-quantized source.
                both = gathered_rows(weight)
                state.ba_weights[id(layer.linear_attn)] = (both[:48], both[48:])
        model._owner_prefill_state = state
        print(
            json.dumps(
                {
                    "owner_prefill_initialized": True,
                    "rank": rank,
                    "ba_replicated_layers": len(state.ba_weights),
                    "workspace_bytes": int(workspace.metadata["buffer_size"]),
                    "native_abi": "owner-prefill-v1",
                }
            ),
            flush=True,
        )
    # Smaller shapes can select Stream-K in the original MLP, whose reduction
    # order differs from the replica kernels. Keep their original TP branches.
    if m >= 2048 and not state.mlp_replicas_prepared:
        prepare_mlp_replicas(state)
        state.mlp_replicas_prepared = True
    assert state.local_mlp_partials is None
    # This is forward-local: never mutate a module-global M between requests.
    state.m = m
    state.p, state.split_rows, state.own, state.other = row_partition(m, state.rank)
    state.local_rows = state.split_rows[state.rank]
    assert (
        0 < state.local_rows <= state.p
        and 0 < state.split_rows[1 - state.rank] <= state.p
    )
    state.verify = (
        state.verified_forwards < envs.VLLM_SM120_OWNER_VERIFY
        and (not envs.VLLM_SM120_OWNER_VERIFY_ONLY_RAGGED or m != 4096)
        and real_prefill(model)
    )
    state.active_checks = []
    state.total_forwards += 1
    return state


def check(state, label, a, b):
    if not state.verify:
        return
    assert a.dtype == b.dtype and a.shape == b.shape, (label, a.shape, b.shape)
    same = bool(
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )
    item = {
        "label": label,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "bitwise_equal": same,
    }
    if not same:
        d = a.float() - b.float()
        item.update(max_abs=float(d.abs().max()), mae=float(d.abs().mean()))
    state.active_checks.append(item)
    if not same:
        print("OWNER_MISMATCH", item, flush=True)


def gather_bytes(state, a, b, *, scale_atom=False):
    """Gather only transport-padded packets, then crop to real model rows."""
    if state.m == 4096:
        out_a = torch.empty((a.numel() * 2,), dtype=torch.uint8, device=a.device)
        out_b = torch.empty((b.numel() * 2,), dtype=torch.uint8, device=b.device)
        torch.ops.mach_owner.gather_mx8(
            a,
            b,
            state.workspace.workspace_tensor,
            out_a,
            out_b,
            state.rank,
            int(state.workspace.metadata["buffer_size"]),
            True,
        )
        return out_a, out_b
    # The new op receives short local input and performs masked zero-tail loads.
    # Its outputs hold both P-row owner packets; only the cropped prefix is a
    # model tensor.  MX8 scales retain 128-row atoms after the value crop.
    if scale_atom:
        value_bytes_per_row = a.numel() // state.local_rows
        scale_bytes_per_row = H // 32
        value_size = 2 * state.p * value_bytes_per_row
        scale_size = 2 * state.p * scale_bytes_per_row
        value_crop = state.m * value_bytes_per_row
        scale_crop = ((state.m + 127) // 128) * 128 * scale_bytes_per_row
    else:
        a_bytes_per_row = a.numel() // state.local_rows
        b_bytes_per_row = b.numel() // state.local_rows
        value_size = 2 * state.p * a_bytes_per_row
        scale_size = 2 * state.p * b_bytes_per_row
        value_crop = state.m * a_bytes_per_row
        scale_crop = state.m * b_bytes_per_row
    out_a = torch.empty((value_size,), dtype=torch.uint8, device=a.device)
    out_b = torch.empty((scale_size,), dtype=torch.uint8, device=b.device)
    torch.ops.mach_owner_ragged.gather_mx8_padded(
        a,
        b,
        state.workspace.workspace_tensor,
        out_a,
        out_b,
        state.rank,
        int(state.workspace.metadata["buffer_size"]),
        True,
    )
    return out_a[:value_crop], out_b[:scale_crop]


def norm_step(state, hidden, residual, norm, label, quantize=True, owner_only=False):
    mxfp6 = import_module("mxfp6")
    oracle_norm: Any = None
    pending = state.local_mlp_partials
    state.local_mlp_partials = None
    if state.verify:
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            fused_allreduce_gemma_rms_norm,
        )

        # These extra full-residual gathers and original kernels run only for
        # explicitly requested verification forwards, outside scored throughput.
        full_residual = padded_rows(state, residual[state.own])
        oracle_partial = pending.oracle_partial if pending is not None else hidden
        oracle_norm, oracle_residual = fused_allreduce_gemma_rms_norm(
            oracle_partial.clone(), full_residual, norm
        )
    output_norm = torch.empty_like(hidden)
    if pending is None:
        if state.m == 4096:
            torch.ops.mach_owner.reduce_owner(
                hidden,
                residual,
                norm.weight,
                state.workspace.workspace_tensor,
                hidden,
                output_norm,
                state.rank,
                int(state.workspace.metadata["buffer_size"]),
                norm.variance_epsilon,
                1.0,
                True,
            )
        else:
            torch.ops.mach_owner_ragged.reduce_owner(
                hidden,
                residual,
                norm.weight,
                state.workspace.workspace_tensor,
                hidden,
                output_norm,
                state.rank,
                state.p,
                int(state.workspace.metadata["buffer_size"]),
                norm.variance_epsilon,
                1.0,
                True,
            )
    else:
        ordered = (
            torch.ops.mach_owner_local.ordered_sum_norm
            if state.m == 4096
            else torch.ops.mach_owner_ragged_local.ordered_sum_norm
        )
        ordered(
            pending.parts[0],
            pending.parts[1],
            residual[state.own],
            norm.weight,
            hidden[state.own],
            output_norm[state.own],
            norm.variance_epsilon,
            1.0,
            True,
        )
    check(
        state,
        label + "/residual",
        hidden[state.own],
        oracle_residual[state.own] if state.verify else hidden[state.own],
    )
    check(
        state,
        label + "/norm",
        output_norm[state.own],
        oracle_norm[state.own] if state.verify else output_norm[state.own],
    )
    prepared = None
    if quantize:
        half = mxfp6.quantize_mxfp8(output_norm[state.own])
        if owner_only:
            prepared = half
        else:
            values, scales = gather_bytes(
                state, half.values, half.scales, scale_atom=True
            )
            prepared = mxfp6.MXFP8Tensor(values.view(state.m, H), scales, state.m, H)
        if state.verify:
            reference = mxfp6.quantize_mxfp8(
                oracle_norm[state.own] if owner_only else oracle_norm
            )
            check(state, label + "/mx8_values", prepared.values, reference.values)
            check(state, label + "/mx8_scales", prepared.scales, reference.scales)
    return SimpleNamespace(
        residual=hidden, norm=output_norm, prepared=prepared, oracle_norm=oracle_norm
    )


def gdn_forward(state, module, normalized, label):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _encode_layer_name,
    )

    qkvz = apply_mxfp8_weight(normalized.prepared, packed_weight(module.in_proj_qkvz))
    normalized.norm[state.other].zero_()
    ba0, ba1 = (
        F.linear(normalized.norm, w)[state.own] for w in state.ba_weights[id(module)]
    )
    both = gather_bytes(state, ba0.view(torch.uint8), ba1.view(torch.uint8))
    ba = both[state.rank].view(torch.bfloat16).view(state.m, 48)
    if state.verify:
        reference_qkvz, _ = module.in_proj_qkvz(normalized.oracle_norm)
        reference_ba, _ = module.in_proj_ba(normalized.oracle_norm)
        check(state, label + "/qkvz", qkvz, reference_qkvz)
        check(state, label + "/ba", ba, reference_ba)
    # Preserve the actual physical M for the original prefill core. Transport
    # packets are cropped before this point; no model tensor is padded.
    core = torch.zeros(
        (state.m, module.num_v_heads // module.tp_size, module.head_v_dim),
        dtype=torch.bfloat16,
        device=qkvz.device,
    )
    torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
        qkvz, ba, core, layer_name=_encode_layer_name(module.prefix)
    )
    output, _ = module.out_proj(core.flatten(-2))
    return output


def attention_forward(state, module, normalized, positions, label):
    qkv = apply_mxfp8_weight(normalized.prepared, packed_weight(module.qkv_proj))
    if state.verify:
        ref, _ = module.qkv_proj(normalized.oracle_norm)
        check(state, label + "/qkv", qkv, ref)
    q, k, v, gate = module._project_qkv_gate(qkv, positions)
    output = module.attn(q, k, v)
    if gate is not None:
        output = output * torch.sigmoid(gate)
    output, _ = module.o_proj(output)
    return output


def run_layer(state, layer, hidden, residual, positions):
    label = f"layer{state.indices[id(layer)]}"
    if residual is None:
        # First input is replicated embedding, not a partial TP sum. Preserve
        # the original whole first input norm and attention invocation.
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        assert layer.layer_type == "linear_attention"
        hidden = layer.linear_attn(hidden_states=normalized)
    else:
        n = norm_step(state, hidden, residual, layer.input_layernorm, label + "/pre")
        residual = n.residual
        hidden = (
            gdn_forward(state, layer.linear_attn, n, label)
            if layer.layer_type == "linear_attention"
            else attention_forward(state, layer.self_attn, n, positions, label)
        )
    replicated = state.m >= 2048 and state.indices[id(layer)] in state.mlp_weights
    n = norm_step(
        state,
        hidden,
        residual,
        layer.post_attention_layernorm,
        label + "/post",
        owner_only=replicated,
    )
    output = (
        local_mlp(state, layer, n, label)
        if replicated
        else packed_mlp(layer.mlp, n.prepared)
    )
    if state.verify and not replicated:
        reference = layer.mlp(n.oracle_norm)
        check(state, label + "/mlp", output, reference)
    return output, n.residual


def finish(state, hidden, residual, norm):
    n = norm_step(state, hidden, residual, norm, "final", quantize=False)
    # One final BF16 all-gather preserves the LM head interface. This cost is
    # present in every candidate service measurement, never repeated per layer.
    output = padded_rows(state, n.norm[state.own])
    check(state, "final/full_bf16", output, n.oracle_norm if state.verify else output)
    if state.verify:
        state.verified_forwards += 1
        state.checks.append(
            {
                "forward": state.verified_forwards,
                "physical_rows": state.m,
                "real_prefill": True,
                "checks": state.active_checks,
            }
        )
        passed = all(c["bitwise_equal"] for f in state.checks for c in f["checks"])
        if not passed:
            raise RuntimeError(
                "Owner prefill shadow verification failed; inspect state.checks"
            )
        print(
            json.dumps(
                {
                    "owner_prefill_verified": True,
                    "rank": state.rank,
                    "verified_forwards": state.verified_forwards,
                    "passed": passed,
                    "checks_this_forward": len(state.active_checks),
                }
            ),
            flush=True,
        )
    return output


def packed_mlp(module, x):
    y = apply_mxfp8_weight(x, packed_weight(module.gate_up_proj))
    return module.down_proj(module.act_fn(y))[0]
