from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_mach.exl3 import mxfp6_checkpoint as checkpoint


def _valid_config() -> dict:
    geometry = dict(checkpoint._GEOMETRY)
    geometry["layer_types"] = checkpoint._expected_layer_types()
    return {
        "text_config": geometry,
        "quantization_config": {
            "quant_method": "quark",
            "export": {"pack_method": "reorder"},
            "global_quant_config": {
                "weight": dict(checkpoint._WEIGHT_QUANT),
                "input_tensors": dict(checkpoint._INPUT_QUANT),
            },
        },
    }


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor], config=None):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(_valid_config() if config is None else config),
        encoding="utf-8",
    )
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard_name)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard_name for name in tensors}}),
        encoding="utf-8",
    )
    return root


def _small_source(key: str, *, fill: int | None = None):
    if fill is None:
        values = torch.arange(16 * 96, dtype=torch.int64).remainder(251)
        values = values.to(torch.uint8).reshape(16, 96)
        scales = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4)
    else:
        values = torch.full((16, 96), fill, dtype=torch.uint8)
        scales = torch.full((16, 4), fill + 2, dtype=torch.uint8)
    return {key + ".weight": values, key + ".weight_scale": scales}


def test_reads_exact_packed_byte_and_logical_scale_slices(tmp_path) -> None:
    key = "model.language_model.layers.0.test_proj"
    tensors = _small_source(key)
    reader = checkpoint.Mxfp6Checkpoint(_write_checkpoint(tmp_path, tensors))
    spec = checkpoint._PartSpec(
        key=key,
        source_rows=16,
        source_k=128,
        row_start=8,
        rows=8,
        k_start=32,
        k=32,
    )

    values, scales = reader._read_part(spec)

    torch.testing.assert_close(values, tensors[key + ".weight"][8:16, 24:48])
    torch.testing.assert_close(scales, tensors[key + ".weight_scale"][8:16, 1:2])
    assert values.is_contiguous()
    assert scales.is_contiguous()


def test_merged_load_preserves_source_order_and_only_packs_scales(
    tmp_path, monkeypatch
) -> None:
    first = "model.language_model.layers.0.first"
    second = "model.language_model.layers.0.second"
    tensors = _small_source(first, fill=1) | _small_source(second, fill=2)
    reader = checkpoint.Mxfp6Checkpoint(_write_checkpoint(tmp_path, tensors))
    specs = tuple(
        checkpoint._PartSpec(key, 16, 128, 0, 8, 0, 128)
        for key in (first, second)
    )
    monkeypatch.setattr(
        checkpoint,
        "_layer_plan",
        lambda prefix, rank: checkpoint._LayerPlan((0, 1), specs, True),
    )
    seen_scales = []

    def pack_scales(logical):
        seen_scales.append(logical.clone())
        return logical.flatten()

    result = reader.load_layer(
        "model.layers.0.mlp.gate_up_proj",
        rank=0,
        tp_size=2,
        device="cpu",
        pack_scales=pack_scales,
    )

    assert result.shard_ids == (0, 1)
    assert result.weights == {}
    assert result.merged_weight is not None
    assert (result.merged_weight.rows, result.merged_weight.k) == (16, 128)
    expected_values = torch.cat(
        [tensors[first + ".weight"][:8], tensors[second + ".weight"][:8]]
    ).flatten()
    torch.testing.assert_close(result.merged_weight.values, expected_values)
    assert torch.all(seen_scales[0][:8] == 3)
    assert torch.all(seen_scales[0][8:] == 4)
    torch.testing.assert_close(result.merged_weight.scales, seen_scales[0].flatten())


def test_nonmerged_load_keeps_the_logical_shard_id(tmp_path, monkeypatch) -> None:
    key = "model.language_model.layers.0.output"
    tensors = _small_source(key, fill=7)
    reader = checkpoint.Mxfp6Checkpoint(_write_checkpoint(tmp_path, tensors))
    spec = checkpoint._PartSpec(key, 16, 128, 0, 8, 0, 128)
    monkeypatch.setattr(
        checkpoint,
        "_layer_plan",
        lambda prefix, rank: checkpoint._LayerPlan((None,), (spec,), False),
    )

    result = reader.load_layer(
        "model.layers.0.mlp.down_proj",
        rank=1,
        tp_size=2,
        device="cpu",
        pack_scales=lambda logical: logical.flatten(),
    )

    assert result.shard_ids == (None,)
    assert result.merged_weight is None
    assert set(result.weights) == {None}
    weight = result.weights[None]
    assert (weight.rows, weight.k) == (8, 128)
    torch.testing.assert_close(weight.values, tensors[key + ".weight"][:8].flatten())


def test_qkv_and_qkvz_tp2_mapping_is_deterministic() -> None:
    linear_rank0 = checkpoint._layer_plan(
        "language_model.model.layers.0.linear_attn.in_proj_qkvz", 0
    )
    linear_rank1 = checkpoint._layer_plan(
        "language_model.model.layers.0.linear_attn.in_proj_qkvz", 1
    )
    assert linear_rank0.shard_ids == (0, 1, 2, 3)
    assert linear_rank0.merged
    assert [(part.row_start, part.rows) for part in linear_rank0.parts] == [
        (0, 1024),
        (2048, 1024),
        (4096, 3072),
        (0, 3072),
    ]
    assert [(part.row_start, part.rows) for part in linear_rank1.parts] == [
        (1024, 1024),
        (3072, 1024),
        (7168, 3072),
        (3072, 3072),
    ]

    full_rank1 = checkpoint._layer_plan(
        "language_model.model.layers.3.self_attn.qkv_proj", 1
    )
    assert full_rank1.shard_ids == ("q", "k", "v")
    assert [(part.row_start, part.rows) for part in full_rank1.parts] == [
        (6144, 6144),
        (512, 512),
        (512, 512),
    ]
    assert all(part.k == 5120 for part in full_rank1.parts)


def test_gate_up_and_row_parallel_mapping() -> None:
    gate_up = checkpoint._layer_plan("model.layers.2.mlp.gate_up_proj", 1)
    assert gate_up.shard_ids == (0, 1)
    assert gate_up.merged
    assert [part.key.rsplit(".", 1)[-1] for part in gate_up.parts] == [
        "gate_proj",
        "up_proj",
    ]
    coordinates = [
        (part.row_start, part.rows, part.k_start, part.k) for part in gate_up.parts
    ]
    assert coordinates == [
        (8704, 8704, 0, 5120),
        (8704, 8704, 0, 5120),
    ]

    down = checkpoint._layer_plan("model.layers.2.mlp.down_proj", 1)
    assert down.shard_ids == (None,)
    assert not down.merged
    assert (down.parts[0].row_start, down.parts[0].rows) == (0, 5120)
    assert (down.parts[0].k_start, down.parts[0].k) == (8704, 8704)


def test_rejects_mismatched_served_geometry(tmp_path) -> None:
    root = _write_checkpoint(tmp_path, _small_source("unused"))
    served = dict(checkpoint._GEOMETRY)
    served["layer_types"] = checkpoint._expected_layer_types()
    served["hidden_size"] = 4096

    with pytest.raises(ValueError, match="served model.*hidden_size=4096"):
        checkpoint.Mxfp6Checkpoint(root, expected_config=SimpleNamespace(**served))


def test_rejects_incompatible_quantization_metadata(tmp_path) -> None:
    config = _valid_config()
    config["quantization_config"]["global_quant_config"]["weight"][
        "group_size"
    ] = 64
    root = _write_checkpoint(tmp_path, _small_source("unused"), config=config)

    with pytest.raises(ValueError, match=r"weight\.group_size=64"):
        checkpoint.Mxfp6Checkpoint(root)


def test_rejects_unsafe_index_filenames(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_valid_config()), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe checkpoint shard filename"):
        checkpoint.Mxfp6Checkpoint(root)


@pytest.mark.parametrize(
    ("rank", "tp_size", "message"),
    [
        (0, 1, "requires TP=2"),
        (2, 2, "rank must be 0 or 1"),
        (True, 2, "rank must be an integer"),
    ],
)
def test_rejects_invalid_tensor_parallel_coordinates(
    tmp_path, rank, tp_size, message
) -> None:
    reader = checkpoint.Mxfp6Checkpoint(
        _write_checkpoint(tmp_path, _small_source("unused"))
    )

    with pytest.raises(ValueError, match=message):
        reader.load_layer(
            "model.layers.0.mlp.down_proj",
            rank=rank,
            tp_size=tp_size,
            device="cpu",
            pack_scales=lambda scales: scales,
        )


def test_rejects_wrong_layer_family_and_unsupported_projection() -> None:
    with pytest.raises(ValueError, match="not a linear-attention layer"):
        checkpoint._layer_plan("model.layers.3.linear_attn.out_proj", 0)
    with pytest.raises(ValueError, match="not a full-attention layer"):
        checkpoint._layer_plan("model.layers.2.self_attn.o_proj", 0)
    with pytest.raises(ValueError, match="unsupported checkpoint projection"):
        checkpoint._layer_plan("model.layers.0.linear_attn.in_proj_a", 0)


def test_rejects_malformed_tensor_shape(tmp_path) -> None:
    key = "model.language_model.layers.0.test_proj"
    tensors = _small_source(key)
    tensors[key + ".weight"] = torch.zeros((16, 95), dtype=torch.uint8)
    reader = checkpoint.Mxfp6Checkpoint(_write_checkpoint(tmp_path, tensors))
    spec = checkpoint._PartSpec(key, 16, 128, 0, 8, 0, 128)

    with pytest.raises(ValueError, match=r"has shape \(16, 95\).+expected \(16, 96\)"):
        reader._read_part(spec)
