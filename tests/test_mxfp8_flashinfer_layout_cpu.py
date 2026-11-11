"""Execute the packaged FlashInfer forward split with actual CPU tensor views."""
import ast
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import textwrap
import types
import unittest

PATCH = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/data/runtime.patch"


def split_fragment():
    lines = PATCH.read_text().splitlines()
    start = lines.index("+        elif self.use_separate_kv_head_slots:")
    end = next(i for i in range(start, len(lines)) if lines[i] == "         else:")
    source = textwrap.dedent("\n".join(line[1:] for line in lines[start:end]
                                         if line.startswith(("+", " "))))
    return source.replace("elif ", "if ", 1)


def actual_forward_fragment(torch):
    dist = metadata.distribution("vllm")
    backend = ast.parse(dist.locate_file("vllm/v1/attention/backends/flashinfer.py").read_text())
    impl = next(n for n in backend.body if isinstance(n, ast.ClassDef) and n.name == "FlashInferImpl")
    forward = next(n for n in impl.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    layout = next(n for n in ast.walk(forward) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "kv_cache_layout" for t in n.targets))
    assert ast.unparse(layout.value) == "attn_metadata.kv_cache_layout"
    utils = ast.parse(dist.locate_file("vllm/v1/attention/backends/utils.py").read_text())
    names = next(n for n in utils.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_FLASHINFER_LAYOUT_NAMES" for t in n.targets))
    getter = next(n for n in utils.body if isinstance(n, ast.FunctionDef)
                  and n.name == "get_flashinfer_layout_string")
    torch_utils = ast.parse(dist.locate_file("vllm/utils/torch_utils.py").read_text())
    canonical = next(n for n in torch_utils.body if isinstance(n, ast.FunctionDef)
                     and n.name == "canonicalize_singleton_dim_strides")
    body = ast.parse("def split(self, attn_metadata, kv_cache_permute):\n    pass").body[0]
    body.body = [layout, *ast.parse(split_fragment()).body,
                 ast.Return(value=ast.Name(id="kv_cache_tuple", ctx=ast.Load()))]
    namespace = {"torch": torch, "KVCacheLayout": object}
    module = ast.Module(body=[names, getter, canonical, body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual-flashinfer-forward-split>", "exec"), namespace)
    return namespace["split"]


class FlashInferLayoutCPU(unittest.TestCase):
    def test_packaged_split_uses_forward_metadata_local(self):
        tree = ast.parse(split_fragment())
        self.assertTrue(any(isinstance(n, ast.Name) and n.id == "kv_cache_layout" for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "kv_cache_layout"
                             for n in ast.walk(tree)))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("vllm"),
                         "requires the target vLLM CPU environment")
    def test_actual_forward_split_nhd_hnd_without_impl_layout_attribute(self):
        import torch
        split = actual_forward_fragment(torch)
        for name, axis in (("LBNHC", 2), ("LBHNC", 1)):
            for heads in (1, 2):
                with self.subTest(layout=name, heads=heads):
                    impl = types.SimpleNamespace(use_separate_kv_head_slots=True, num_kv_heads=heads)
                    self.assertFalse(hasattr(impl, "kv_cache_layout"))
                    attn_metadata = types.SimpleNamespace(kv_cache_layout=types.SimpleNamespace(name=name))
                    shape = (1, 5, 2 * heads, 8) if axis == 2 else (1, 2 * heads, 5, 8)
                    cache = torch.arange(__import__("math").prod(shape), dtype=torch.float32).to(torch.bfloat16).reshape(shape)
                    actual = split(impl, attn_metadata, cache)
                    expected = cache.split(heads, dim=axis)
                    self.assertEqual(len(actual), 2)
                    for found, wanted in zip(actual, expected):
                        self.assertEqual(found.shape, wanted.shape)
                        self.assertTrue(torch.equal(found, wanted))
                        self.assertEqual(found.untyped_storage().data_ptr(), cache.untyped_storage().data_ptr())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
