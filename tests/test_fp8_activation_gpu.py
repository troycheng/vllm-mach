"""GPU qualification of the Mach group128 mutation and graph boundary.

Synthetic inputs test operator contracts, not serving performance or model
accuracy. Root runs this suite on the frozen service image after native build.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires the SM120 qualification GPU"
)


@pytest.fixture(scope="module")
def activation_op():
    from vllm import _custom_ops  # noqa: F401
    from vllm_mach.fp8 import activation

    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_MACH_FP8_SILU", "1")
        activation.register()
        activation.verify_runtime()
        activation.install()
        yield activation, torch.ops.vllm_mach_fp8.silu_and_mul_per_block_quant.default


def allocate(x, group=128, transposed=True, output_dtype=None):
    rows, width = x.shape
    hidden = width // 2
    out = torch.empty(
        (rows, hidden), device=x.device,
        dtype=output_dtype or torch.float8_e4m3fn,
    )
    shape = (hidden // group, rows) if transposed else (rows, hidden // group)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    return out, scales.t() if transposed else scales


def assert_exact(reference, actual):
    assert torch.equal(reference[0].view(torch.uint8), actual[0].view(torch.uint8))
    assert torch.equal(reference[1].view(torch.int32), actual[1].view(torch.int32))


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                                  256, 512, 1024, 2048])
@pytest.mark.parametrize("transposed", [False, True])
@torch.inference_mode()
def test_native_preserves_fp8_bytes_and_scale_bits(activation_op, rows, transposed):
    activation, op = activation_op
    generator = torch.Generator(device="cuda").manual_seed(1729 + rows)
    x = torch.randn((rows, 18432), generator=generator, device="cuda",
                    dtype=torch.bfloat16)
    expected = allocate(x, transposed=transposed)
    actual = allocate(x, transposed=transposed)
    before = activation.inspect()["counts"].get("native", 0)
    torch.ops._C.silu_and_mul_per_block_quant(
        expected[0], x, expected[1], 128, None, transposed
    )
    op(actual[0], x, actual[1], 128, None, transposed)
    assert activation.inspect()["counts"].get("native", 0) == before + 1
    assert_exact(expected, actual)


@pytest.mark.parametrize("dtype,group,width,upper", [
    (torch.float16, 128, 18432, None),
    (torch.bfloat16, 64, 18432, None),
    (torch.bfloat16, 128, 1024, None),
    (torch.bfloat16, 128, 18432, 0.001),
])
@torch.inference_mode()
def test_nonqualified_contract_uses_original_op(
    activation_op, dtype, group, width, upper
):
    activation, op = activation_op
    x = torch.randn((3, width), device="cuda", dtype=dtype)
    ub = None if upper is None else torch.tensor(
        upper, device="cuda", dtype=torch.float32
    )
    expected, actual = allocate(x, group=group), allocate(x, group=group)
    before = activation.inspect()["counts"].get("fallback", 0)
    torch.ops._C.silu_and_mul_per_block_quant(
        expected[0], x, expected[1], group, ub, True
    )
    op(actual[0], x, actual[1], group, ub, True)
    assert activation.inspect()["counts"].get("fallback", 0) == before + 1
    assert_exact(expected, actual)


@pytest.mark.parametrize("rows", [3, 32, 64, 2048])
@torch.inference_mode()
def test_changing_input_graph_replay_overwrites_output(activation_op, rows):
    activation, op = activation_op
    x = torch.randn((rows, 18432), device="cuda", dtype=torch.bfloat16)
    actual = allocate(x)
    expected = allocate(x)
    for _ in range(3):
        op(actual[0], x, actual[1], 128, None, True)
    torch.cuda.synchronize()
    before = activation.inspect()["counts"].get("capture_native", 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(actual[0], x, actual[1], 128, None, True)
    assert activation.inspect()["counts"].get("capture_native", 0) > before
    previous = actual[0].view(torch.uint8).clone()
    for change in (0.25, -0.5, 1.0):
        x.add_(change)
        actual[0].view(torch.uint8).fill_(0x7F)
        actual[1].fill_(float("nan"))
        torch.ops._C.silu_and_mul_per_block_quant(
            expected[0], x, expected[1], 128, None, True
        )
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(expected, actual)
        assert not torch.equal(previous, actual[0].view(torch.uint8))
        previous.copy_(actual[0].view(torch.uint8))


def test_auto_functionalized_and_fake_preserve_mutation_contract(activation_op):
    from torch._higher_order_ops.auto_functionalize import auto_functionalized

    activation, op = activation_op
    x = torch.randn((3, 18432), device="cuda", dtype=torch.bfloat16)
    out, scales = allocate(x)
    out.view(torch.uint8).fill_(0x7F)
    scales.fill_(float("nan"))
    before = activation.inspect()["counts"].get("native", 0)
    result = auto_functionalized(
        op, out=out, input=x, scales=scales, group_size=128,
        scale_ub=None, is_scale_transposed=True,
    )
    assert result[0] is None
    assert activation.inspect()["counts"].get("native", 0) == before + 1
    expected = allocate(x)
    torch.ops._C.silu_and_mul_per_block_quant(
        expected[0], x, expected[1], 128, None, True
    )
    assert_exact(expected, result[1:])
    assert torch.isnan(scales).all()
    assert (out.view(torch.uint8) == 0x7F).all()
    torch.library.opcheck(
        op, (out, x, scales, 128, None, True),
        test_utils=("test_faketensor",),
    )


def test_float8_schema_checker_limitation_is_shared_by_original(activation_op):
    """Torch2.13 schema randomization cannot multiply a Float8 mutable out."""
    _, op = activation_op
    x = torch.randn((3, 18432), device="cuda", dtype=torch.bfloat16)
    for target in (torch.ops._C.silu_and_mul_per_block_quant.default, op):
        out, scales = allocate(x)
        result = torch.library.opcheck(
            target, (out, x, scales, 128, None, True),
            test_utils=("test_schema",), raise_exception=False,
        )
        error = str(result["test_schema"])
        assert "mul_cuda" in error and "Float8_e4m3fn" in error


def test_dynamic_compiled_call_preserves_bytes_input_and_output_lifetime(
    activation_op,
):
    """Exercise actual AOT dispatch without the unsupported Float8 randomizer."""
    from torch._higher_order_ops.auto_functionalize import auto_functionalized

    activation, op = activation_op
    def functional(x):
        out, scales = allocate(x)
        result = auto_functionalized(
            op, out=out, input=x, scales=scales, group_size=128,
            scale_ub=None, is_scale_transposed=True,
        )
        return result[1], result[2]

    compiled = torch.compile(functional, dynamic=True, fullgraph=True)
    retained = []
    for rows in (3, 32, 64, 3):
        x = torch.randn((rows, 18432), device="cuda", dtype=torch.bfloat16)
        original_input = x.view(torch.int16).clone()
        before = activation.inspect()["counts"].get("native", 0)
        actual = compiled(x)
        assert activation.inspect()["counts"].get("native", 0) > before
        expected = allocate(x)
        torch.ops._C.silu_and_mul_per_block_quant(
            expected[0], x, expected[1], 128, None, True
        )
        assert_exact(expected, actual)
        assert torch.equal(original_input, x.view(torch.int16))
        for earlier, snapshot in retained:
            assert_exact(snapshot, earlier)
        retained.append((actual, tuple(t.clone() for t in actual)))
