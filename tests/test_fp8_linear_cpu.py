# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for dispatch, layer ownership, and graph allocation boundaries."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1] / "src/vllm_mach/fp8"


def load(monkeypatch, filename):
    name = "fp8_test_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, shape, dtype="e4m3", *, strides=None, contiguous=True,
                 device=None, payload=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = device or types.SimpleNamespace(type="cuda", index=0)
        self.is_cuda = self.device.type == "cuda"
        self.contiguous = contiguous
        self.payload = payload
        if strides is None:
            strides = (self.shape[1], 1) if self.ndim == 2 else (1,) * self.ndim
        self.strides = strides

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def is_contiguous(self):
        return self.contiguous

    def repeat_interleave(self, repeats, dim):
        assert (repeats, dim) == (2, 0)
        return Tensor((self.shape[0] * 2, self.shape[1]), self.dtype,
                      device=self.device, payload=self.payload)

    def new_empty(self, shape, dtype=None):
        shape = (shape,) if isinstance(shape, int) else shape
        return Tensor(shape, dtype or self.dtype, device=self.device)

    @property
    def T(self):
        return Tensor(self.shape[::-1], self.dtype, device=self.device)

    def view(self, *shape):
        if -1 in shape:
            shape = (self.numel() // shape[1], shape[1])
        return Tensor(shape, self.dtype, device=self.device, payload=self.payload)

    def __add__(self, bias):
        return Tensor(self.shape, self.dtype, device=self.device,
                      payload=("bias", bias))

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def to(self, *, dtype):
        assert dtype == self.dtype
        return self


@pytest.fixture
def backend(monkeypatch):
    events = []
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.float8_e4m3fn, torch.float32, torch.bfloat16 = "e4m3", "f32", "bf16"
    torch.cuda = types.SimpleNamespace(
        current_device=lambda: 0, get_device_capability=lambda index: (12, 0),
        is_current_stream_capturing=lambda: False,
    )
    torch.empty = lambda shape, **kwargs: Tensor(shape, kwargs["dtype"],
                                               device=kwargs["device"])
    registrations = []

    class Op:
        def __init__(self, impl):
            self.impl = impl
        def __call__(self, *args):
            return self.impl(*args)
        def register_fake(self, fake):
            self.fake = fake

    def custom_op(name, impl, **kwargs):
        registrations.append((name, kwargs))
        return Op(impl)
    torch.library = types.SimpleNamespace(custom_op=custom_op)
    monkeypatch.setitem(sys.modules, "torch", torch)
    linear = load(monkeypatch, "linear.py")
    native = types.ModuleType("vllm_mach.fp8.block_native")
    native.verify_runtime = lambda: events.append("native_verify")
    native.gemm = lambda a, b, sa, sb64: (
        events.append(("n64", sb64)) or Tensor((a.shape[0], b.shape[0]), "bf16")
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    ordered = types.ModuleType("vllm_mach.fp8.ordered")
    ordered.gemm = lambda a, sa, b, sb, out, raw: (
        events.append(("ordered", out, raw)) or out
    )
    monkeypatch.setitem(sys.modules, ordered.__name__, ordered)
    ops = types.ModuleType("vllm._custom_ops")
    ops.cutlass_scaled_mm = lambda a, b, **kwargs: (
        events.append("stock_op") or Tensor((a.shape[0], b.shape[1]), "bf16")
    )
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    class Cutlass:
        config = types.SimpleNamespace(out_dtype="bf16")
        apply_input_quant = True
        use_triton = False
        def process_weights_after_loading(self, layer):
            events.append("stock_prepare")
        def apply_weights(self, layer, x, bias=None, **kwargs):
            events.append("stock_apply")
            return "stock_apply"
        def _get_layer_params(self, layer):
            return types.SimpleNamespace(weight=layer.weight,
                                         input_scale=None, input_scale_ub=None)
        def quant_fp8(self, x, scale, upper_bound, **kwargs):
            events.append(("stock_quant", kwargs))
            m, k = x.shape
            return (Tensor((m, k)),
                    Tensor((m, k // 128), "f32", strides=(1, m)))
    cutlass = types.ModuleType("vllm.model_executor.kernels.linear.scaled_mm.cutlass")
    cutlass.CutlassFp8BlockScaledMMKernel = Cutlass
    monkeypatch.setitem(sys.modules, cutlass.__name__, cutlass)
    return linear, torch, Cutlass, events, registrations


class Layer:
    def __init__(self, payload):
        self.weight = Tensor((2560, 4096))
        self.weight_scale = Tensor((20, 32), "f32", payload=payload)
        self.persistent = {}
    def register_buffer(self, name, value, *, persistent):
        setattr(self, name, value)
        self.persistent[name] = persistent


def inputs(m, k=4096):
    return (Tensor((m, k)), Tensor((2560, k)),
            Tensor((m, k // 128), "f32", strides=(1, m)),
            Tensor((20, k // 128), "f32"),
            Tensor((40, k // 128), "f32"))


def test_import_and_register_do_not_initialize_cuda_or_load_native(backend):
    linear, torch, _, events, registrations = backend
    assert linear.torch is None
    torch.cuda = types.SimpleNamespace()
    linear.register()
    linear.register()
    assert events == []
    assert len(registrations) == 1
    assert registrations[0][1]["mutates_args"] == ()


@pytest.mark.parametrize("m,route", [(1, "ordered"), (8, "ordered"),
                                     (9, "stock"), (16, "n64"),
                                     (128, "n64"), (136, "stock")])
def test_runtime_m_routes_stay_within_the_qualified_boundary(backend, m, route):
    linear, _, _, _, _ = backend
    linear.verify_runtime(n64=False)
    assert linear.select_route(*inputs(m)) == route


def test_e8m0_other_devices_and_nonqualified_scale_layouts_fall_back(backend):
    linear, _, _, _, _ = backend
    linear.verify_runtime(n64=False)
    for index in (2, 3, 4):
        values = list(inputs(32))
        values[index].dtype = "ue8m0"
        assert linear.select_route(*values) == "stock"
    values = list(inputs(32))
    values[0].device = types.SimpleNamespace(type="cuda", index=1)
    assert linear.select_route(*values) == "stock"
    values = list(inputs(32))
    values[2].strides = (32, 1)
    assert linear.select_route(*values) == "stock"


def test_a_b_switches_are_independent_and_qualified_errors_are_visible(backend):
    linear, _, _, _, _ = backend
    linear.install()
    assert linear.select_route(*inputs(4), ordered=False) == "stock"
    assert linear.select_route(*inputs(32), n64=False) == "stock"
    native = sys.modules["vllm_mach.fp8.block_native"]
    def failed(*args):
        raise RuntimeError("qualified launch failed")
    native.gemm = failed
    with pytest.raises(RuntimeError, match="qualified launch failed"):
        linear.gemm(*inputs(32))


def test_shared_kernel_uses_each_layers_owned_scale_and_reload_refreshes(backend):
    linear, _, Cutlass, events, _ = backend
    original_prepare, original_apply = (Cutlass.process_weights_after_loading,
                                        Cutlass.apply_weights)
    linear.install()
    assert not linear.install()
    first, second, kernel = Layer("first"), Layer("second"), Cutlass()
    for layer in (first, second):
        kernel.process_weights_after_loading(layer)
    assert first._mach_block_fp8_scale64 is not second._mach_block_fp8_scale64
    for layer in (first, second):
        output = kernel.apply_weights(layer, Tensor((2, 16, 4096), "bf16"))
        assert output.shape == (2, 16, 2560)
    assert [event[1].payload for event in events
            if isinstance(event, tuple) and event[0] == "n64"] == ["first", "second"]
    assert first.persistent["_mach_block_fp8_scale64"] is False
    previous = first._mach_block_fp8_scale64
    first.weight_scale.payload = "reloaded"
    kernel.process_weights_after_loading(first)
    assert first._mach_block_fp8_scale64 is not previous
    assert first._mach_block_fp8_scale64.payload == "reloaded"
    assert linear.inspect()["derived_scale_layers"] == 2
    linear.uninstall()
    assert Cutlass.process_weights_after_loading is original_prepare
    assert Cutlass.apply_weights is original_apply


def test_capture_refresh_is_refused_before_mutating_buffer(backend):
    linear, torch, Cutlass, _, _ = backend
    linear.install()
    layer = Layer("scale")
    Cutlass().process_weights_after_loading(layer)
    previous = layer._mach_block_fp8_scale64
    torch.cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="before graph capture"):
        linear.refresh_scales(layer)
    assert layer._mach_block_fp8_scale64 is previous


def test_original_quant_bias_reshape_and_static_fallback_remain_intact(backend):
    linear, _, Cutlass, events, _ = backend
    linear.install()
    layer, kernel = Layer("scale"), Cutlass()
    kernel.process_weights_after_loading(layer)
    bias = object()
    output = kernel.apply_weights(layer, Tensor((2, 16, 4096), "bf16"), bias)
    assert output.shape == (2, 16, 2560)
    assert output.payload == ("bias", bias)
    assert ("stock_quant", {"use_triton": False}) in events
    events.clear()
    layer.weight_scale.dtype = "ue8m0"
    assert kernel.apply_weights(layer, Tensor((32, 4096), "bf16")) == "stock_apply"
    assert events == ["stock_apply"]


def test_reload_to_unsupported_layout_drops_old_derived_scale(backend):
    linear, _, Cutlass, _, _ = backend
    linear.install()
    layer, kernel = Layer("scale"), Cutlass()
    kernel.process_weights_after_loading(layer)
    layer.weight = Tensor((4096, 4096))
    kernel.process_weights_after_loading(layer)
    assert not hasattr(layer, "_mach_block_fp8_scale64")


def test_ordered_invocations_allocate_independent_output_and_scratch(backend):
    linear, _, _, events, _ = backend
    linear.install(n64=False, ordered=True)
    first, second = linear.gemm(*inputs(4)), linear.gemm(*inputs(4))
    assert first is not second
    launches = [event for event in events
                if isinstance(event, tuple) and event[0] == "ordered"]
    assert launches[0][2] is not launches[1][2]
    assert launches[0][2].shape == (32, 4, 2560)
    assert "native_verify" not in events


def test_fake_preserves_symbolic_m_without_reading_it(backend):
    linear, _, _, _, _ = backend
    class Symbol:
        def __int__(self):
            raise AssertionError("symbolic M was materialized")
    m = Symbol()
    result = linear._fake(*inputs(m), True, True)
    assert result.shape == (m, 2560)


def test_missing_binary_and_wrong_schema_fail_before_native_is_usable(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0+cu130"
    torch.ops = types.SimpleNamespace(load_library=lambda path: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    native = load(monkeypatch, "block_native.py")
    library = tmp_path / "missing.so"
    monkeypatch.setenv("VLLM_MACH_BLOCK_FP8_LIBRARY", str(library))
    with pytest.raises(ImportError, match="does not exist"):
        native.verify_runtime()
    library.write_bytes(b"test-binary")
    torch.ops.vllm_mach_block_fp8 = types.SimpleNamespace(mm=types.SimpleNamespace(
        default=types.SimpleNamespace(_schema="mm(Tensor x) -> Tensor")))
    with pytest.raises(RuntimeError, match="schema mismatch"):
        native.verify_runtime()
    assert native._LOADED is False
    with pytest.raises(RuntimeError, match="before model compilation"):
        native.gemm(*inputs(32)[:3], inputs(32)[4])
