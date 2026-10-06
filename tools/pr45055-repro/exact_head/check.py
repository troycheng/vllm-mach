"""Bounded CUDA contracts for the exact PR kernel/private stable registration."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized


def allocate(x, dtype, group, transposed):
    rows, input_width = x.shape
    hidden = input_width // 2
    out = torch.empty((rows, hidden), dtype=dtype, device=x.device)
    shape = (hidden // group, rows) if transposed else (rows, hidden // group)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    return out, scales.t() if transposed else scales


def differences(expected, actual):
    return {
        "quantized_bytes_changed": int((expected[0].view(torch.uint8)
                                        != actual[0].view(torch.uint8)).sum()),
        "scale_bits_changed": int((expected[1].view(torch.int32)
                                   != actual[1].view(torch.int32)).sum()),
    }


def require_exact(expected, actual):
    result = differences(expected, actual)
    if any(result.values()):
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--safety-only", action="store_true")
    args = parser.parse_args()
    from vllm import _custom_ops  # noqa: F401

    root = Path(__file__).resolve().parent
    libraries = list(root.glob("pr45055_exact_ext*.so"))
    if len(libraries) != 1:
        raise RuntimeError("Expected exactly one built private extension")
    library, = libraries
    torch.ops.load_library(str(library))
    op = torch.ops.pr45055_exact.run.default
    torch.library.register_fake("pr45055_exact::run", lambda *args, **kwargs: None)
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This qualification plan is scoped to SM120")
    record = {
        "build": json.loads((root / "build_identity.json").read_text()),
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "cuda_only": True, "complete": False, "cases": [], "graphs": [],
        "limitations": ["ROCm unqualified", "no model/service qualification",
                        "exact-head int32 token-offset multiplication risk"],
        "int32_offset_example": {
            "token_index": 65536, "input_stride": 32768,
            "int64_expected": 2147483648, "int32_wrapped": -2147483648,
            "evidence": "source/CPU arithmetic, not CUDA overflow execution",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        rows = [3, 64] if args.safety_only else [1, 3, 32, 64, 2048]
        for m, dtype, quant, group, transposed in itertools.product(
            rows, [torch.bfloat16, torch.float16],
            [torch.float8_e4m3fn, torch.int8], [64, 128], [False, True],
        ):
            x = torch.randn((m, 18432), device="cuda", dtype=dtype)
            expected = allocate(x, quant, group, transposed)
            actual = allocate(x, quant, group, transposed)
            torch.ops._C.silu_and_mul_per_block_quant(
                expected[0], x, expected[1], group, None, transposed
            )
            op(actual[0], x, actual[1], group, None, transposed)
            torch.cuda.synchronize()
            record["cases"].append({
                "rows": m, "input_dtype": str(dtype), "quant_dtype": str(quant),
                "group_size": group, "transposed": transposed,
                **require_exact(expected, actual),
            })
            save()
        for group, groups in itertools.product([64, 128], [3, 5]):
            x = torch.randn((3, 2 * group * groups), device="cuda",
                            dtype=torch.bfloat16)
            ub = torch.tensor(0.001, device="cuda", dtype=torch.float32)
            expected = allocate(x, torch.float8_e4m3fn, group, True)
            actual = allocate(x, torch.float8_e4m3fn, group, True)
            torch.ops._C.silu_and_mul_per_block_quant(
                expected[0], x, expected[1], group, ub, True
            )
            op(actual[0], x, actual[1], group, ub, True)
            record["cases"].append({
                "rows": 3, "groups": groups, "group_size": group,
                "upper_bound": 0.001, **require_exact(expected, actual),
            })
            save()
        for m in (32, 64):
            x = torch.randn((m, 18432), device="cuda", dtype=torch.bfloat16)
            actual = allocate(x, torch.float8_e4m3fn, 128, True)
            expected = allocate(x, torch.float8_e4m3fn, 128, True)
            for _ in range(3):
                op(actual[0], x, actual[1], 128, None, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op(actual[0], x, actual[1], 128, None, True)
            for step in range(3):
                x.add_(0.25)
                torch.ops._C.silu_and_mul_per_block_quant(
                    expected[0], x, expected[1], 128, None, True
                )
                graph.replay()
                torch.cuda.synchronize()
                record["graphs"].append({
                    "rows": m, "step": step, **require_exact(expected, actual),
                })
                save()
        out, scales = actual
        result = auto_functionalized(
            op, out=out, input=x, scales=scales, group_size=128,
            scale_ub=None, is_scale_transposed=True,
        )
        require_exact(expected, result[1:])
        torch.library.opcheck(
            op, (out, x, scales, 128, None, True), test_utils=("test_faketensor",),
        )
        record["schema_checker_float8_limitation"] = {}
        for name, target in (("original", torch.ops._C.silu_and_mul_per_block_quant.default),
                             ("exact_head", op)):
            errors = torch.library.opcheck(
                target, (out, x, scales, 128, None, True),
                test_utils=("test_schema",), raise_exception=False,
            )
            error = str(errors["test_schema"])
            assert "mul_cuda" in error and "Float8_e4m3fn" in error
            record["schema_checker_float8_limitation"][name] = error
        def functional(input_):
            quantized, sf = allocate(input_, torch.float8_e4m3fn, 128, True)
            result = auto_functionalized(
                op, out=quantized, input=input_, scales=sf, group_size=128,
                scale_ub=None, is_scale_transposed=True,
            )
            return result[1], result[2]
        compiled = torch.compile(functional, dynamic=True, fullgraph=True)
        for rows in (3, 32, 64):
            x = torch.randn((rows, 18432), device="cuda", dtype=torch.bfloat16)
            input_bits = x.view(torch.int16).clone()
            actual = compiled(x)
            expected = allocate(x, torch.float8_e4m3fn, 128, True)
            torch.ops._C.silu_and_mul_per_block_quant(
                expected[0], x, expected[1], 128, None, True
            )
            require_exact(expected, actual)
            assert torch.equal(input_bits, x.view(torch.int16))
        record["auto_functionalized_fake_and_actual_dynamic_compile"] = "passed"
        record["complete"] = True
        save()
    except Exception as exc:
        record["error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
