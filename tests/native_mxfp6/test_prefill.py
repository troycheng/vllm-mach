# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TP2 transport and graph regressions from the optimization series."""

from contextlib import contextmanager

import pytest
import torch
from torch.multiprocessing import spawn
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _can_use_flashinfer,
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@contextmanager
def ensure_current_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def init_test_distributed_environment(tp_size, pp_size, rank, port, local_rank):
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=tp_size * pp_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=local_rank,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)


@pytest.mark.parametrize(
    "shape,dtype,tp,expected",
    [
        ((4096, 5120), torch.bfloat16, 2, "direct"),
        ((3001, 5120), torch.bfloat16, 2, "long"),
        ((32, 5120), torch.bfloat16, 2, None),
        ((3072, 5120), torch.bfloat16, 2, None),
        ((4096, 4096), torch.bfloat16, 2, None),
        ((4096, 5120), torch.float16, 2, None),
        ((4096, 5120), torch.bfloat16, 4, None),
    ],
)
def test_lossless_prefill_restricts_native_shape_contract(shape, dtype, tp, expected):
    from vllm_mach.mxfp6.sm120_lossless_prefill import _codec_kind

    assert _codec_kind(shape, dtype, tp) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 4),
        ("tp_rank", 1),
        ("hidden_dim", 4096),
        ("max_token_num", 4095),
        ("buffer_size", 84213759),
    ],
)
def test_lossless_prefill_rejects_incompatible_workspace(field, value):
    from types import SimpleNamespace

    from vllm_mach.mxfp6.sm120_lossless_prefill import _validate_workspace

    metadata = dict(
        tp_size=2, tp_rank=0, hidden_dim=5120, max_token_num=6553, buffer_size=84213760
    )
    metadata[field] = value
    with pytest.raises(RuntimeError, match="compatible TP2"):
        _validate_workspace(
            SimpleNamespace(backend="trtllm", metadata=metadata), 0, 4096
        )


def test_lossless_prefill_reports_missing_optional_dependency(monkeypatch):
    import importlib.metadata

    from vllm_mach.mxfp6 import sm120_lossless_prefill as codec

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    codec._load_codec.cache_clear()
    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="requires flashinfer-python"):
        codec._load_codec("direct")


@ensure_current_vllm_config()
def _worker_lossless_prefill(local_rank, port, graph_codec):
    import os

    from vllm_mach.mxfp6 import sm120_lossless_prefill as codec

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(2, 1, local_rank, port, local_rank=local_rank)
    set_random_seed(981)
    norm = GemmaRMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.normal_(0, 0.1)
    os.environ["VLLM_SM120_LOSSLESS_PREFILL_VERIFY"] = "1"
    os.environ["VLLM_SM120_LOSSLESS_PREFILL_GRAPH"] = str(int(graph_codec))

    # Alternate decode graphs and eager prefill on the same IPC workspace.
    # Includes odd row ownership, changing inputs, raw-codec fallbacks and
    # in-place residual output; reference is installed FlashInfer, bitwise.
    from unittest.mock import patch

    if graph_codec:
        cold = torch.empty(1024, 5120, device=device, dtype=torch.bfloat16)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            pytest.raises(RuntimeError, match="eager warmup"),
        ):
            codec.try_lossless_prefill(
                cold, torch.empty_like(cold), norm, torch.empty_like(cold), 2, 6553
            )

    for rows, capture in (
        (32, True),
        (4096, False),
        (3000, False),
        (3001, False),
        (3890, False),
        (4096, False),
        (3000, True),
        (4096, True),
        (32, True),
    ):
        partial = torch.randn(rows, 5120, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(partial)
        partial[0, :4] = torch.tensor(
            [0.0, -0.0, 1e-20, 1e20], device=device, dtype=torch.bfloat16
        )
        os.environ["VLLM_SM120_LOSSLESS_PREFILL"] = "0"
        ref, ref_res = fused_allreduce_gemma_rms_norm(partial.clone(), residual, norm)
        assert _can_use_flashinfer(partial, 2)[0]
        os.environ["VLLM_SM120_LOSSLESS_PREFILL"] = "1"
        original_residual = residual.clone()
        x = partial.clone()
        if capture:
            graph = torch.cuda.CUDAGraph()
            # Capture must never lazily load a library. Track cached codec calls
            # separately so a silent FlashInfer fallback cannot pass this test.
            from unittest.mock import Mock

            tracked = {k: Mock(wraps=v) for k, v in codec._LOADED_CODECS.items()}
            with (
                patch.dict(codec._LOADED_CODECS, tracked),
                patch.object(
                    codec,
                    "_load_codec",
                    side_effect=AssertionError("lazy load in graph"),
                ),
                torch.cuda.graph(graph),
            ):
                out, res = fused_allreduce_gemma_rms_norm(x, residual, norm)
            expected_calls = int(graph_codec and rows != 32)
            assert sum(v.call_count for v in tracked.values()) == expected_calls
            x.copy_(partial)
            graph.replay()
        else:
            out, res = fused_allreduce_gemma_rms_norm(x, residual, norm)
            assert rows in codec._VERIFIED[norm]
        torch.accelerator.synchronize()
        assert res.data_ptr() == x.data_ptr()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
        assert torch.equal(res.view(torch.int16), ref_res.view(torch.int16))
        assert torch.equal(
            residual.view(torch.int16), original_residual.view(torch.int16)
        )
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="SM120 required",
)
@pytest.mark.parametrize("graph_codec", [False, True])
def test_lossless_prefill_matches_flashinfer_across_decode_graphs(graph_codec):
    import importlib.util

    if current_platform.device_count() < 2:
        pytest.skip("TP2 requires two GPUs")
    if any(
        importlib.util.find_spec(f"mach_lossless_prefill_{kind}_ext") is None
        for kind in ("direct", "long")
    ):
        pytest.skip("Optional lossless-prefill CUDA extensions not installed")
    spawn(
        _worker_lossless_prefill,
        args=(str(get_open_port()), graph_codec),
        nprocs=2,
        join=True,
    )


@ensure_current_vllm_config()
def _worker_lossless_raw_graph(local_rank, port):
    """Exercise the native codec itself, independent of the eager-only dispatcher."""
    import os

    from vllm.distributed import get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    from vllm_mach.mxfp6 import sm120_lossless_prefill as codec

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(2, 1, local_rank, port, local_rank=local_rank)
    os.environ["VLLM_SM120_LOSSLESS_PREFILL"] = "0"
    set_random_seed(781)
    norm = GemmaRMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.normal_(0, 0.1)
    graphs = {}
    retained = {32, 1024, 3001, 3890, 4096}
    for rows in [32, *sorted(codec._LONG_ROWS), 4096]:
        x = torch.zeros(rows, 5120, device=device, dtype=torch.bfloat16)
        residual = torch.zeros_like(x)
        output = torch.empty_like(x)
        ok, budget = _can_use_flashinfer(x, 2)
        assert ok
        ws = get_fi_ar_workspace(
            world_size=2,
            rank=local_rank,
            max_token_num=budget,
            hidden_dim=5120,
            dtype=x.dtype,
            group=get_tp_group().cpu_group,
        )
        kind = codec._codec_kind(tuple(x.shape), x.dtype, 2)
        run = None
        if kind is not None:
            codec._validate_workspace(ws, local_rank, rows)
            run = codec._load_codec(kind)

        def call(x=x, residual=residual, output=output, ws=ws, kind=kind, run=run):
            if kind is None:
                return fused_allreduce_gemma_rms_norm(x, residual, norm)
            args = (
                x,
                residual,
                norm.weight,
                ws.workspace_tensor,
                x,
                output,
                local_rank,
                int(ws.metadata["buffer_size"]),
                1e-6,
                1.0,
            )
            if kind == "direct":
                run(*args, 1, True)
            else:
                run(*args, True)
            return output, x

        # Load libraries and initialize all resources before capture.
        call()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, res = call()
        for fixture in range(4):
            torch.manual_seed(1400 + fixture)
            residual.normal_()
            norm.weight.data.normal_(0, 0.1)
            torch.manual_seed(1700 + fixture + local_rank)
            partial = torch.randn_like(x)
            if fixture == 1:
                partial.zero_()
                partial[:, ::2] = -0.0
            elif fixture == 2:
                partial[0, :6] = torch.tensor(
                    [0.0, -0.0, 1e-30, 1e20, float("inf"), float("nan")],
                    device=device,
                    dtype=x.dtype,
                )
            elif fixture == 3:
                partial.mul_(0.0001)
            ref, ref_res = fused_allreduce_gemma_rms_norm(
                partial.clone(), residual, norm
            )
            before = residual.clone()
            x.copy_(partial)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
                rows,
                fixture,
                "norm",
            )
            assert torch.equal(res.view(torch.int16), ref_res.view(torch.int16)), (
                rows,
                fixture,
                "residual",
            )
            assert torch.equal(residual.view(torch.int16), before.view(torch.int16))
        if rows in retained:
            graphs[rows] = (graph, x, residual, out, res)
        print(f"RAW_CODEC_GRAPH_PASS rank={local_rank} M={rows} fixtures=4", flush=True)

    # Replay previously captured shapes in changing order, sharing the same
    # Lamport/barrier workspace with ordinary decode and intervening eager calls.
    for iteration, rows in enumerate([4096, 32, 3001, 3890, 1024, 32, 4096] * 3):
        graph, x, residual, out, res = graphs[rows]
        torch.manual_seed(2500 + iteration)
        residual.normal_()
        torch.manual_seed(2800 + iteration + local_rank)
        partial = torch.randn_like(x)
        ref, ref_res = fused_allreduce_gemma_rms_norm(partial.clone(), residual, norm)
        x.copy_(partial)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
            iteration,
            rows,
        )
        assert torch.equal(res.view(torch.int16), ref_res.view(torch.int16)), (
            iteration,
            rows,
        )
    print(f"RAW_CODEC_GRAPH_MIXED_PASS rank={local_rank} replays=21", flush=True)
    del graphs, graph
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="SM120 required",
)
def test_lossless_native_codec_graph_replay_all_supported_shapes():
    import importlib.util

    if current_platform.device_count() < 2:
        pytest.skip("TP2 requires two GPUs")
    if any(
        importlib.util.find_spec(f"mach_lossless_prefill_{kind}_ext") is None
        for kind in ("direct", "long")
    ):
        pytest.skip("Optional lossless-prefill CUDA extensions not installed")
    spawn(_worker_lossless_raw_graph, args=(str(get_open_port()),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "rows",
    [
        512,
        513,
        768,
        1024,
        1025,
        1536,
        2048,
        2049,
        2560,
        2999,
        3000,
        3001,
        3023,
        4095,
        4096,
    ],
)
def test_owner_prefill_partition_preserves_every_real_row(rows):
    from vllm_mach.mxfp6.sm120_owner_prefill import row_partition

    p0, counts0, own0, other0 = row_partition(rows, 0)
    p1, counts1, own1, other1 = row_partition(rows, 1)
    assert p0 == p1 and p0 % 128 == 0
    assert counts0 == counts1 and sum(counts0) == rows
    assert all(0 < n <= p0 for n in counts0)
    assert own0.start == 0 and own1.stop == rows and own0.stop == own1.start
    assert other0 == own1 and other1 == own0


@pytest.mark.parametrize("raw", ["[0,0]", "[64]", "[-1]", "[true]", "[1.5]", "{}"])
def test_owner_prefill_rejects_invalid_replica_selection(monkeypatch, raw):
    from vllm_mach.mxfp6.sm120_owner_prefill import replica_layers

    monkeypatch.setenv("VLLM_SM120_OWNER_MLP_LAYERS", raw)
    with pytest.raises(ValueError, match="distinct layer indices"):
        replica_layers()


@pytest.mark.parametrize("rows", [32, 511, 1024, 1052, 4097])
def test_owner_prefill_fallback_does_not_initialize_owner_resources(monkeypatch, rows):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_mach.mxfp6 import sm120_owner_prefill as owner

    monkeypatch.setenv("VLLM_SM120_OWNER_PREFILL", "1")
    monkeypatch.setenv("VLLM_SM120_LOSSLESS_PREFILL", "1")
    load = Mock(side_effect=AssertionError("owner initialized during decode"))
    monkeypatch.setattr(owner, "load_native", load)
    assert owner.begin(SimpleNamespace(), torch.zeros(rows, 5120)) is None
    load.assert_not_called()


@ensure_current_vllm_config()
def _worker_owner_norm(local_rank, port):
    from types import SimpleNamespace

    import flashinfer.comm
    from vllm.distributed import get_tp_group

    from vllm_mach.mxfp6 import sm120_owner_prefill as owner

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(2, 1, local_rank, port, local_rank=local_rank)
    owner.load_native()
    workspace = flashinfer.comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=2,
        rank=local_rank,
        max_token_num=4128,
        hidden_dim=5120,
        dtype=torch.bfloat16,
        group=get_tp_group().cpu_group,
    )
    norm = GemmaRMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    for rows in (
        512,
        513,
        768,
        1024,
        1025,
        1536,
        2048,
        2049,
        2560,
        2999,
        3000,
        3001,
        3023,
        4095,
        4096,
    ):
        torch.manual_seed(rows)
        norm.weight.data.normal_(0, 0.1)
        residual = torch.randn(rows, 5120, dtype=torch.bfloat16, device=device)
        torch.manual_seed(rows + local_rank)
        partial = torch.randn_like(residual)
        p, counts, own, other = owner.row_partition(rows, local_rank)
        state = SimpleNamespace(
            m=rows,
            p=p,
            rank=local_rank,
            own=own,
            other=other,
            local_rows=counts[local_rank],
            workspace=workspace,
            verify=True,
            local_mlp_partials=None,
            active_checks=[],
        )
        result = owner.norm_step(state, partial, residual, norm, f"M{rows}")
        assert len(state.active_checks) == 4
        assert all(c["bitwise_equal"] for c in state.active_checks), state.active_checks
        gathered = owner.padded_rows(state, result.norm[own])
        assert torch.equal(
            gathered.view(torch.int16), result.oracle_norm.view(torch.int16)
        )
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="SM120 required",
)
def test_owner_prefill_ragged_norm_and_transport_match_flashinfer():
    import importlib.util

    if current_platform.device_count() < 2:
        pytest.skip("TP2 requires two GPUs")
    if importlib.util.find_spec("mach_owner_prefill_ext") is None:
        pytest.skip("Optional owner-prefill extensions not installed")
    spawn(_worker_owner_norm, args=(str(get_open_port()),), nprocs=2, join=True)
