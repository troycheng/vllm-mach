from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_mach.mxfp6 import warmup
from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel


class _Layer(torch.nn.Module):
    def __init__(self, output_features: int = 8, input_features: int = 128):
        super().__init__()
        self.scheme = SimpleNamespace(
            ocp_mx_linear=object.__new__(Mxfp6Sm120LinearKernel)
        )
        self.weight = torch.nn.Parameter(
            torch.empty((output_features, input_features * 3 // 4), dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.empty((output_features, input_features // 32), dtype=torch.uint8),
            requires_grad=False,
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = _Layer()
        self.duplicate = _Layer()


def test_collect_w6a8_problems_deduplicates_shapes() -> None:
    problems = warmup._collect_w6a8_problems(_Model())
    assert len(problems) == 1
    assert problems[0][:2] == (8, 128)


def test_warmup_plans_workspace_and_normalizes_sizes(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class Runtime:
        @staticmethod
        def PackedMXFP6Tensor(**kwargs):
            return SimpleNamespace(**kwargs)

        @staticmethod
        def begin_workspace_planning(device):
            calls.append(("begin", device))

        @staticmethod
        def warmup_w6a8(x, weight, *, out_dtype, iterations):
            calls.append(("warm", (x.shape[0], weight.rows, weight.k, out_dtype)))
            assert iterations == 1

        @staticmethod
        def finalize_workspace_planning(device):
            calls.append(("finalize", device))

    monkeypatch.setattr(warmup, "_import_mxfp6", lambda: Runtime)
    monkeypatch.setattr(warmup.torch.cuda, "synchronize", lambda device: None)

    warmup.warmup_mxfp6_sm120(_Model(), [4, 16, 4, 0], torch.float32)

    assert [name for name, _ in calls] == ["begin", "warm", "warm", "finalize"]
    assert calls[1][1] == (16, 8, 128, torch.bfloat16)
    assert calls[2][1] == (4, 8, 128, torch.bfloat16)


def test_capture_stream_warmup_stops_after_registering_lane(monkeypatch) -> None:
    calls: list[int] = []

    class Runtime:
        lanes = 1

        @staticmethod
        def PackedMXFP6Tensor(**kwargs):
            return SimpleNamespace(**kwargs)

        @classmethod
        def warmup_w6a8(cls, x, weight, *, out_dtype, iterations):
            del weight, out_dtype
            assert iterations == 1
            calls.append(x.shape[0])
            cls.lanes += 1

        @classmethod
        def workspace_stats(cls, device):
            del device
            return {"lanes": cls.lanes}

    monkeypatch.setattr(warmup, "_import_mxfp6", lambda: Runtime)
    monkeypatch.setattr(warmup.torch.cuda, "synchronize", lambda device: None)

    warmup.warmup_mxfp6_sm120_stream(_Model(), [16, 4], torch.bfloat16)

    assert calls == [4]
