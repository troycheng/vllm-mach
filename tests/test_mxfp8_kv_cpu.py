"""CPU scale-schema and worker lifecycle checks with explicit device fakes."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/kv.py"
spec = importlib.util.spec_from_file_location("mach_kv_cpu_test", PATH)
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)


def metadata():
    return {"profile": kv.PROFILE, "kv_scales": [
        {"layer_index": i, "module_name": f"model.layers.{i}.self_attn.attn",
         "k_scale": .125, "v_scale": .25} for i in kv.LAYERS
    ]}


class SchemaCPU(unittest.TestCase):
    def test_explicit_selection_hash_and_dtype_restrictions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "scales.json"
            source = metadata()
            path.write_text(json.dumps(source))
            self.assertEqual(kv.selection()["kv_scale_source"], "model")
            os.environ[kv.SCALES_ENV] = str(path)
            first = kv.selection()
            source["kv_scales"][0]["k_scale"] = .5
            path.write_text(json.dumps(source))
            self.assertNotEqual(first["kv_scale_profile_sha256"], kv.selection()["kv_scale_profile_sha256"])
            os.environ[kv.DTYPE_ENV] = "bfloat16"
            with self.assertRaisesRegex(ValueError, "Explicit KV scales"):
                kv.selection()
            del os.environ[kv.SCALES_ENV]
            self.assertIsNone(kv.selection()["kv_scale_source"])
            os.environ[kv.DTYPE_ENV] = "auto"
            with self.assertRaises(ValueError):
                kv.selected_dtype()

    def test_mapping_file_directory_and_default_model_directory(self):
        source = metadata()
        self.assertEqual(kv.load_profile(source), source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mach_profile.json"
            path.write_text(json.dumps(source))
            for profile in (path, directory, None):
                self.assertEqual(kv.load_profile(profile, model_dir=directory), source)
        unordered = copy.deepcopy(source)
        unordered["kv_scales"].reverse()
        self.assertEqual(kv.load_profile(unordered), source)

    def test_wrong_coverage_names_and_scale_numbers_fail(self):
        variants = []
        invalid = metadata(); invalid["profile"] = "other"; variants.append(invalid)
        invalid = metadata(); invalid["kv_scales"].pop(); variants.append(invalid)
        invalid = metadata(); invalid["kv_scales"][1] = invalid["kv_scales"][0]; variants.append(invalid)
        invalid = metadata(); invalid["kv_scales"][0]["module_name"] = "wrong"; variants.append(invalid)
        for value in (True, "0.125", 0., -1., float("nan"), float("inf"), .1, 1.e100):
            invalid = metadata(); invalid["kv_scales"][0]["k_scale"] = value; variants.append(invalid)
        for profile in variants:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                kv.load_profile(profile)


class Device:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.split(":")[0]
    def __eq__(self, other):
        return self.value == str(other)
    def __str__(self):
        return self.value


class Tensor:
    def __init__(self, value=1., *, dtype="f32", device="cpu", shape=()):
        self.value, self.dtype, self.device, self.shape = value, dtype, Device(device), shape
    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total
    def fill_(self, value):
        self.value = value
        return self
    def item(self):
        return self.value
    def stride(self):
        return (1,) * len(self.shape)
    def data_ptr(self):
        return id(self)


class FlashInferImpl:
    def __init__(self):
        self.cache_dtype = "fp8_e4m3"
        self.num_heads, self.num_kv_heads, self.head_size = 16, 4, 256
        self.bmm1_scale = self.bmm2_scale = None


def worker_and_modules():
    torch = types.ModuleType("torch")
    torch.Tensor, torch.device = Tensor, Device
    torch.float32, torch.uint8, torch.float8_e4m3fn, torch.bfloat16 = "f32", "u8", "e4m3", "bf16"
    torch.tensor = lambda value, *, dtype, device: Tensor(value, dtype=dtype, device=device)
    torch.no_grad = nullcontext
    torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
    modules = {}
    for row in metadata()["kv_scales"]:
        module = types.SimpleNamespace(impl=FlashInferImpl(), kv_cache_dtype="fp8_e4m3",
                    kv_cache=Tensor(dtype="u8", device="cuda:0", shape=(0,)))
        for limb in ("k", "v"):
            setattr(module, "_" + limb + "_scale", Tensor(device="cuda:0"))
            # Reproduce the wrong default-device CPU mirror from the old path.
            setattr(module, "_" + limb + "_scale_cpu", Tensor(device="cuda:0"))
            setattr(module, "_" + limb + "_scale_float", 1.)
        modules[row["module_name"]] = module
    model = types.SimpleNamespace(named_modules=lambda: list(modules.items()) + [("other", object())])
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(cache_dtype="fp8_e4m3"),
                                model_config=types.SimpleNamespace(model="unused-model-dir"))
    worker = types.SimpleNamespace(device="cuda:0", vllm_config=cfg,
                                   model_runner=types.SimpleNamespace(model=model))
    backend = types.ModuleType("vllm.v1.attention.backends.flashinfer")
    backend.FlashInferImpl = FlashInferImpl
    return worker, modules, torch, backend


class LifecycleCPU(unittest.TestCase):
    def test_bf16_storage_and_no_fp8_scale_installation(self):
        worker, modules, torch, backend = worker_and_modules()
        worker.vllm_config.cache_config.cache_dtype = "bfloat16"
        for module in modules.values():
            module.impl.cache_dtype = module.kv_cache_dtype = "bfloat16"
            module.kv_cache = Tensor(dtype="bf16", device="cuda:0", shape=(2, 16, 4, 256))
        with self.dependencies(torch, backend):
            receipt = kv.inspect_cache(worker)
            self.assertEqual(len(receipt["layers"]), 8)
            self.assertFalse(receipt["installed"])
            first = next(iter(modules.values()))
            first.kv_cache.dtype = "u8"
            with self.assertRaisesRegex(RuntimeError, "BF16 KV storage"):
                kv.inspect_cache(worker)
            first.kv_cache.dtype = "bf16"
            setattr(worker, kv._STATE, {})
            with self.assertRaisesRegex(RuntimeError, "must not install"):
                kv.inspect_cache(worker)

    def dependencies(self, torch, backend):
        return patch.dict(sys.modules, {"torch": torch,
                    "vllm.v1.attention.backends.flashinfer": backend})

    def test_preprofile_mirrors_and_bound_storage_inspection(self):
        worker, modules, torch, backend = worker_and_modules()
        first = next(iter(modules.values()))
        old_device_owner = first._k_scale
        with self.dependencies(torch, backend):
            receipt = kv.install_scales(worker, metadata())
            self.assertEqual(len(receipt["layers"]), 8)
            self.assertIs(first._k_scale, old_device_owner)
            self.assertEqual(first._k_scale_cpu.device.type, "cpu")
            self.assertEqual(first._k_scale_float, .125)
            self.assertEqual(first._v_scale_float, .25)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                kv.inspect_scales(worker)
            for module in modules.values():
                module.kv_cache = Tensor(dtype="u8", device="cuda:0", shape=(2, 16, 4, 256))
                module.impl.bmm2_scale = .25
            receipt = kv.inspect_scales(worker)
            self.assertTrue(receipt["require_storage"])
            self.assertTrue(all(row["kv_dtype"] == "u8" for row in receipt["layers"]))
            first.impl.bmm2_scale = 1.
            with self.assertRaisesRegex(RuntimeError, "cached V"):
                kv.inspect_scales(worker)
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                kv.install_scales(worker, metadata())

    def test_late_cache_failure_leaves_all_scales_unchanged(self):
        worker, modules, torch, backend = worker_and_modules()
        list(modules.values())[-1].impl.bmm1_scale = 1.
        with self.dependencies(torch, backend):
            with self.assertRaisesRegex(RuntimeError, "cache initialized"):
                kv.install_scales(worker, metadata())
            self.assertFalse(hasattr(worker, kv._STATE))
            self.assertTrue(all(module._k_scale.item() == 1. for module in modules.values()))
            self.assertTrue(all(module._k_scale_cpu.device.type == "cuda" for module in modules.values()))

    def test_coverage_geometry_dtype_and_capture_fail_before_writes(self):
        for mutation in (
            lambda worker, modules, torch: modules.pop(next(iter(modules))),
            lambda worker, modules, torch: setattr(next(iter(modules.values())).impl, "head_size", 128),
            lambda worker, modules, torch: setattr(worker.vllm_config.cache_config, "cache_dtype", "auto"),
            lambda worker, modules, torch: setattr(torch.cuda, "is_current_stream_capturing", lambda: True),
        ):
            worker, modules, torch, backend = worker_and_modules()
            mutation(worker, modules, torch)
            with self.dependencies(torch, backend), self.assertRaises(RuntimeError):
                kv.install_scales(worker, metadata())
            self.assertFalse(hasattr(worker, kv._STATE))

    def test_mirror_value_drift_is_fatal(self):
        worker, modules, torch, backend = worker_and_modules()
        with self.dependencies(torch, backend):
            kv.install_scales(worker, metadata())
            next(iter(modules.values()))._k_scale_float = 1.
            with self.assertRaisesRegex(RuntimeError, "mirrors changed"):
                kv.inspect_scales(worker, require_storage=False)


if __name__ == "__main__":
    unittest.main()


def test_selected_4b_mxfp8_hybrid_capacity():
    from importlib.metadata import version
    import pytest

    if version("vllm").split("+")[0] != "0.34.0":
        pytest.skip("vLLM 0.34 changed the default cache block size")

    import torch
    from vllm.config.cache import CacheConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

    shapes = ((3, 8192), (32, 128, 128))
    dtypes = (torch.bfloat16, torch.float32)
    model_cls = types.SimpleNamespace(get_mamba_state_shape_from_config=lambda cfg: shapes,
                                      get_mamba_state_dtype_from_config=lambda cfg: dtypes)
    model = types.SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration", dtype=torch.bfloat16, max_model_len=8192,
        get_head_size=lambda: 256, get_num_kv_heads=lambda pc: 4, get_num_attention_heads=lambda pc: 16,
        is_hybrid=True, use_mla=False,
    )
    four_b = "qwen35-4b-mxfp8-champion-v1"
    two_b = "qwen35-2b-mxfp8-champion-v1"
    for profile, explicit, page, blocks, tokens in (
        (four_b, None, 1056, 1179, 878033), (four_b, 64, 1088, 1144, 851968),
        (two_b, None, 1088, 1144, 851968), ("", None, 1088, 1144, 851968),
    ):
        cache = CacheConfig(block_size=explicit, cache_dtype="fp8_e4m3",
                            mamba_ssm_cache_dtype="float32")
        cache.kv_cache_layout = "LBHNC"
        cfg = types.SimpleNamespace(model_config=model, cache_config=cache, use_flashinfer_replayssm=False,
                                    attention_config=types.SimpleNamespace(hisparse_config=None),
                                    parallel_config=types.SimpleNamespace(decode_context_parallel_size=1))
        with patch.dict(os.environ, {"VLLM_MACH_PROFILE": profile}), \
             patch.object(Platform, "_find_non_ssm_backends", return_value=[FlashInferBackend]), \
             patch.object(ModelRegistry, "resolve_model_cls", return_value=(model_cls, "")):
            Platform.update_block_size_for_backend(cfg)
        assert cache.block_size == page
        attn = FullAttentionSpec(block_size=page, num_kv_heads=4, head_size=256, dtype=torch.float8_e4m3fn)
        ssm = MambaSpec(block_size=8192, shapes=shapes, dtypes=dtypes,
                        page_size_padded=cache.mamba_page_size_padded)
        assert attn.page_size_bytes == ssm.page_size_bytes == page * 2048
        groups = [KVCacheGroupSpec([f"layer_{g}_{i}" for i in range(8)], spec)
                  for g, spec in enumerate((ssm, ssm, ssm, attn))]
        plan = get_kv_cache_config_from_groups(cfg, groups, 19 * 2**30)
        assert plan.num_blocks == blocks
        assert get_kv_cache_capacity(cfg, plan)[0] == tokens
