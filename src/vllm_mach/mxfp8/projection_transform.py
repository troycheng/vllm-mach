# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, line-preserving QKVZ/BA AOT call-site rewrite.

Only executable assignments in Runner.call are changed. The large generated
compile-time docstring, Triton sources, launch metadata, and every other line
are preserved verbatim. The runtime dispatch must check alignment of the raw
weight arguments passed before their original copy_if_misaligned calls.
"""

import ast
import hashlib
import json
from typing import Any


IMPORT = "from vllm_mach.mxfp8.projection_parallel import dispatch as _qkvz_ba_pair"
Q = "torch.ops.vllm.mach_mx8_dual_qkvz_both.default"
B = "torch.ops.mx8_ba_only.small_n.default"
ALLOWED_ASSERTS = {"assert_size_stride", "assert_alignment"}


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return base + "." + node.attr if base else None
    return None


def _name(node: ast.AST) -> str:
    if not isinstance(node, ast.Name):
        raise ValueError("call arguments and assignment targets must be plain Names")
    return node.id


def _call_assignment(stmt: ast.stmt) -> tuple[str, str, tuple[str, ...], int | None] | None:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    if not isinstance(stmt.value, ast.Call):
        return None
    symbol = _dotted(stmt.value.func)
    if symbol not in (Q, B):
        return None
    if stmt.value.keywords or len(stmt.value.args) != (4 if symbol == Q else 3):
        raise ValueError(f"unexpected {symbol} signature at line {stmt.lineno}")
    target = _name(stmt.targets[0])
    args = tuple(_name(arg) for arg in stmt.value.args[:3])
    if symbol == Q:
        lid = stmt.value.args[3]
        if not isinstance(lid, ast.Constant) or type(lid.value) is not int or not 0 <= lid.value <= 23:
            raise ValueError(f"QKVZ layer ID is not a literal 0..23 at line {stmt.lineno}")
        return ("Q", target, args, lid.value)
    return ("B", target, args, None)


def _triton_hash(tree: ast.Module) -> tuple[str, int]:
    # Captures the generated multiline source literals, without interpreting
    # their comments, nested quotes, or Python surface spelling.
    values = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and ("import triton" in node.value or "@triton.jit" in node.value)]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return _sha(payload), len(values)


def _runner_call(tree: ast.Module) -> ast.FunctionDef:
    runners = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Runner"]
    if len(runners) != 1:
        raise ValueError(f"expected one Runner class; got {len(runners)}")
    calls = [node for node in runners[0].body if isinstance(node, ast.FunctionDef) and node.name == "call"]
    if len(calls) != 1:
        raise ValueError(f"expected one Runner.call; got {len(calls)}")
    return calls[0]


def _argument_names(fn: ast.FunctionDef) -> set[str]:
    # Current generated Runner.call unpacks the AOT flat arg list once.
    unpack = [stmt for stmt in fn.body if isinstance(stmt, ast.Assign)
              and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Tuple)
              and isinstance(stmt.value, ast.Name) and stmt.value.id == "args"]
    if len(unpack) != 1:
        raise ValueError("cannot prove AOT argument origin from one flat unpack")
    return {_name(node) for node in unpack[0].targets[0].elts}


def _target_suites(fn: ast.FunctionDef) -> list[tuple[list[ast.stmt], list[tuple[int, tuple[str, str, tuple[str, ...], int | None]]]]]:
    found = []

    def visit(suite: list[ast.stmt]) -> None:
        hits = [(index, match) for index, stmt in enumerate(suite)
                if (match := _call_assignment(stmt)) is not None]
        if hits:
            found.append((suite, hits))
        for stmt in suite:
            for field, value in ast.iter_fields(stmt):
                if field in ("body", "orelse", "finalbody") and isinstance(value, list):
                    visit(value)
                elif field == "handlers" and isinstance(value, list):
                    for handler in value:
                        if isinstance(handler, ast.ExceptHandler):
                            visit(handler.body)

    visit(fn.body)
    return found


def _intermediate(stmts: list[ast.stmt], input_name: str,
                  deferred_weights: set[str], arg_names: set[str]) -> list[str]:
    late = []
    for stmt in stmts:
        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if _name(target) == input_name or _name(target) in deferred_weights:
                    raise ValueError(f"deferred input deleted between calls at line {stmt.lineno}")
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if (_dotted(stmt.value.func) not in ALLOWED_ASSERTS
                    or any(isinstance(node, ast.Call) and node is not stmt.value
                           for node in ast.walk(stmt.value))):
                raise ValueError(f"non-assert call between projections at line {stmt.lineno}")
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = _name(stmt.targets[0])
            if target == input_name:
                raise ValueError(f"shared input reassigned at line {stmt.lineno}")
            if isinstance(stmt.value, ast.Name):
                # Existing generated bufX = bufY aliases are retained.
                if target in deferred_weights:
                    raise ValueError(f"deferred weight redefined at line {stmt.lineno}")
                continue
            if (isinstance(stmt.value, ast.Call)
                    and _dotted(stmt.value.func) == "copy_if_misaligned"
                    and len(stmt.value.args) == 1 and not stmt.value.keywords
                    and _name(stmt.value.args[0]) == target
                    and target in deferred_weights and target in arg_names):
                late.append(target)
                continue
        raise ValueError(f"non-whitelisted statement between calls at line {stmt.lineno}")
    return late


def transform_source(text: str) -> tuple[str, dict[str, Any]]:
    tree = ast.parse(text)
    fn = _runner_call(tree)
    arg_names = _argument_names(fn)
    original_triton = _triton_hash(tree)
    suites = _target_suites(fn)
    if not suites:
        raise ValueError("Runner.call contains no target projection calls")
    lines = text.splitlines(keepends=True)
    edits: dict[int, str] = {}
    entries: list[dict[str, Any]] = []
    skipped_layer_ids: list[int] = []
    used_ids: set[int] = set()
    for suite, hits in suites:
        if len(hits) % 2:
            raise ValueError("unpaired QKVZ/BA call in one executable suite")
        for first_hit, second_hit in zip(hits[::2], hits[1::2]):
            i, first = first_hit
            j, second = second_hit
            if {first[0], second[0]} != {"Q", "B"}:
                raise ValueError("target calls overlap or do not alternate")
            q = first if first[0] == "Q" else second
            b = first if first[0] == "B" else second
            if (q[2][0] != b[2][0] or q[1] == b[1]
                    or {q[1], b[1]} & set((*q[2], *b[2]))):
                raise ValueError("pair does not share one input or has aliased outputs")
            lid = q[3]
            assert lid is not None
            if lid in used_ids:
                raise ValueError(f"repeated QKVZ layer {lid}")
            used_ids.add(lid)
            if any(name not in arg_names for name in (*q[2][1:], *b[2][1:])):
                raise ValueError("weight/scale is not a top-level AOT argument")
            if lid == 0:
                # Layer 0 has executable BA split/clone GPU work between these
                # calls. Keep its complete original sequence and source bytes.
                skipped_layer_ids.append(0)
                continue
            late = _intermediate(suite[i+1:j], q[2][0], set(second[2][1:]), arg_names)
            first_stmt, second_stmt = suite[i], suite[j]
            if not isinstance(first_stmt, ast.Assign) or not isinstance(second_stmt, ast.Assign):
                raise ValueError("internal assignment mismatch")
            if first_stmt.lineno != first_stmt.end_lineno or second_stmt.lineno != second_stmt.end_lineno:
                raise ValueError("projection assignment must occupy one physical line")
            pending = f"_qkvz_ba_pending_{lid}"
            if pending in text:
                raise ValueError("generated pending name already exists")
            indent1 = lines[first_stmt.lineno-1][:len(lines[first_stmt.lineno-1])-len(lines[first_stmt.lineno-1].lstrip())]
            indent2 = lines[second_stmt.lineno-1][:len(lines[second_stmt.lineno-1])-len(lines[second_stmt.lineno-1].lstrip())]
            if indent1 != indent2:
                raise ValueError("projection calls are at different indentation")
            qout, bout = (q[1], pending) if first[0] == "Q" else (pending, b[1])
            x, qw, qs = q[2]
            _, bw, bp = b[2]
            args = f"{x}, {qw}, {qs}, {lid}, {bw}, {bp}, {first[0] == 'Q'}"
            edits[first_stmt.lineno] = f"{indent1}{qout}, {bout} = _qkvz_ba_pair({args})\n"
            edits[second_stmt.lineno] = f"{indent2}{second[1]} = {pending}\n"
            entries.append({"layer_id": lid, "q_line": suite[i if first[0] == 'Q' else j].lineno,
                            "ba_line": suite[i if first[0] == 'B' else j].lineno,
                            "first": first[0], "input": x, "q_output": q[1],
                            "ba_output": b[1], "late_alignment_weights": late})
    if not entries and not skipped_layer_ids:
        raise ValueError("no complete projection pair found")
    # Guard against target expressions outside Runner.call, including future
    # modules with a second call in another method or unexpected nesting.
    all_hits = sum(_call_assignment(node) is not None for node in ast.walk(tree)
                   if isinstance(node, ast.Assign))
    if all_hits != 2*(len(entries) + len(skipped_layer_ids)):
        raise ValueError("target call exists outside paired Runner.call assignments")
    if any(_dotted(node.func) in (Q, B) for node in ast.walk(tree)
           if isinstance(node, ast.Call) and not any(node is stmt.value for suite, _ in suites
                                                    for stmt in suite if isinstance(stmt, ast.Assign))):
        raise ValueError("target call exists outside simple assignments")
    if "_qkvz_ba_pair" in text:
        raise ValueError("runtime dispatch helper name already exists")
    if not entries:
        return text, {"status": "original_order_preserved", "original_sha256": _sha(text),
                      "transformed_sha256": _sha(text), "triton_string_sha256": original_triton[0],
                      "triton_string_count": original_triton[1],
                      "docstring_sha256": _sha(ast.get_docstring(tree) or ""),
                      "import_before_line": None, "changed_call_lines": [],
                      "pairs": [], "skipped_layer_ids": skipped_layer_ids}
    first_exec = next((node.lineno for node in tree.body
                       if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                               and isinstance(node.value.value, str))), None)
    if first_exec is None:
        raise ValueError("module has no executable import location")
    futures = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "__future__"]
    if futures:
        last_future = max(node.lineno for node in futures)
        if any(node.lineno < last_future for node in tree.body
               if node.lineno >= first_exec and node not in futures):
            raise ValueError("unexpected future import order")
        first_exec = last_future + 1
    output = []
    for lineno, line in enumerate(lines, 1):
        if lineno == first_exec:
            output.append(IMPORT + "\n")
        output.append(edits.get(lineno, line))
    changed = "".join(output)
    revised = ast.parse(changed)
    if _triton_hash(revised) != original_triton:
        raise ValueError("Triton source literals changed")
    if ast.get_docstring(tree) != ast.get_docstring(revised):
        raise ValueError("compile-time module docstring changed")
    if len(edits) != 2*len(entries):
        raise ValueError("duplicate physical edit line")
    audit = {"status": "transformed", "original_sha256": _sha(text),
             "transformed_sha256": _sha(changed), "triton_string_sha256": original_triton[0],
             "triton_string_count": original_triton[1], "docstring_sha256": _sha(ast.get_docstring(tree) or ""),
             "import_before_line": first_exec, "changed_call_lines": sorted(edits),
             "pairs": sorted(entries, key=lambda row: row["layer_id"]),
             "skipped_layer_ids": skipped_layer_ids}
    return changed, audit
