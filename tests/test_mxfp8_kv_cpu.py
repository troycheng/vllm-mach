"""CPU scale-schema and worker lifecycle checks with explicit device fakes."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.util
import json
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
    torch.float32, torch.uint8, torch.float8_e4m3fn = "f32", "u8", "e4m3"
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
