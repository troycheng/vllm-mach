#!/usr/bin/env python3
"""Read-only smoke check with external B12X imports and version lookup blocked.

Run in a fresh process with CUDA_VISIBLE_DEVICES selecting an idle SM120 GPU.
This does not uninstall packages or alter the production launcher's checks.
"""

import importlib.abc
import json
import os
import sys
from importlib import metadata


def main():
    attempts = []

    class NoB12x(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "b12x" or fullname.startswith("b12x."):
                attempts.append(fullname)
                raise ModuleNotFoundError("External B12X deliberately blocked")

    assert not any(n == "b12x" or n.startswith("b12x.") for n in sys.modules)
    sys.meta_path.insert(0, NoB12x())
    original_version = metadata.version

    def version(name):
        if name.lower().replace("_", "-") == "b12x":
            raise metadata.PackageNotFoundError(name)
        return original_version(name)

    metadata.version = version
    os.environ["VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS"] = "32"
    os.environ["VLLM_HYBRID_NVFP4_LM_HEAD_USE_FLASHINFER_TOPK"] = "1"
    import mxfp6
    import torch

    from vllm_mach.mxfp6 import hybrid_nvfp4_lm_head as head
    from vllm_mach.mxfp6.sm120_owner_prefill_native import load

    mxfp6.load_library()
    load()
    with torch.inference_mode():
        torch.manual_seed(20260916)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        )
        assert head.prepare_hybrid_nvfp4_lm_head(layer, candidates=128)
        state = head.get_hybrid_nvfp4_lm_head(layer)
        hidden = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)

        def run():
            coarse = state.coarse_logits(hidden, None)
            candidates = state.select_candidates(coarse)
            return candidates, state.refine_logits(
                hidden, layer.weight, candidates, None
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            indices, values = run()
        hidden.normal_()
        graph.replay()
        expected = torch.nn.functional.linear(hidden, layer.weight).gather(
            1, indices.long()
        )
        torch.testing.assert_close(values, expected, atol=0.25, rtol=0.01)
        assert torch.isfinite(values).all()
    assert not any(n == "b12x" or n.startswith("b12x.") for n in sys.modules)
    print(
        json.dumps(
            {
                "status": "PASS",
                "blocked_import_attempts": attempts,
                "checks": [
                    "NVFP4 preparation",
                    "FlashInfer B12X GEMM",
                    "FlashInfer top-k",
                    "BF16 refinement",
                    "changing-input CUDA graph",
                    "MXFP6 library loading",
                    "owner-prefill library loading",
                ],
                "scope": "Small-shape API smoke test, not an uninstalled-package full-model serving run",
            }
        )
    )


if __name__ == "__main__":
    main()
