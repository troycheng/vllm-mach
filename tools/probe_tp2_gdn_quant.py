#!/usr/bin/env python3
"""Reject/accept the extension-owned GDN producer before model integration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import triton
import mxfp6
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260916)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('gdn_quant_candidate', args.producer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = mxfp6.load_library()
    torch.manual_seed(args.seed)
    results = []
    for m in (1, 2, 4, 8, 16, 24, 32):
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(m, 24, 128, device='cuda', dtype=torch.bfloat16)
            storage = torch.randn(m, 8192, device='cuda', dtype=torch.bfloat16)
            z = storage[:, 5120:].view(m, 24, 128)
            w = torch.randn(128, device='cuda', dtype=dtype)
            def reference():
                y = layer_norm_fwd(x.view(-1, 128), w, None, 1.e-6,
                    z=z.reshape(-1, 128), is_rms_norm=True, norm_before_gate=True,
                    activation='silu')[0]
                q, s = torch.ops.mxfp6.quantize_mxfp8(y.view(m, 3072))
                return q, s, y.view_as(x)
            # Changing-input captured replay and extremes, including padding.
            module.produce(x, z, w)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = module.produce(x, z, w)
            checks = []
            for factor in (0., 1.e-20, 1.e-5, 1., 10., 100.):
                x.normal_().mul_(factor)
                storage.normal_().mul_(10. if factor == 100. else 1.)
                actual[1].fill_(213)
                graph.replay()
                expected = reference()
                # Independent FP64 arithmetic reference for the producer.
                xd, zd = x.double(), z.double()
                oracle = (xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1.e-6)
                          * w.double() * (zd * torch.sigmoid(zd)))
                checks.append(dict(factor=factor,
                    norm_mismatches=int((actual[2].view(torch.int16) != expected[2].view(torch.int16)).sum()),
                    code_mismatches=int((actual[0] != expected[0]).sum()),
                    scale_mismatches=int((actual[1] != expected[1]).sum()),
                    oracle_max_abs=float((actual[2].double()-oracle).abs().max())))
            base = [triton.testing.do_bench_cudagraph(reference) for _ in range(5)]
            fused = [triton.testing.do_bench_cudagraph(lambda: module.produce(x,z,w)) for _ in range(5)]
            results.append(dict(rows=m, weight_dtype=str(dtype), checks=checks,
                reference_ms=base, candidate_ms=fused))
            print(m, dtype, checks, base, fused, flush=True)
    payload = dict(schema='mach-tp2-gdn-quant-probe/v1', seed=args.seed,
        extension_library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(args.producer.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
    main()
