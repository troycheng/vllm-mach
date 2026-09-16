#!/usr/bin/env python3
"""Test the existing FlashInfer packed-group producer against Mach's MXFP8 contract.

This is a two-rank numerical compatibility probe, never a serving benchmark.
"""
import argparse
import json
from pathlib import Path
import tempfile


def worker(rank, init, output):
    import torch
    import torch.distributed as dist
    import mxfp6
    from flashinfer.comm import create_allreduce_fusion_workspace, allreduce_fusion
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2)
    workspace=create_allreduce_fusion_workspace(backend='trtllm',world_size=2,rank=rank,
        max_token_num=32,hidden_dim=5120,dtype=torch.bfloat16,group=dist.group.WORLD)
    results=[]
    for m in (1,2,4,8,16,24,32):
        for magnitude in (0.0,1e-20,1.0):
            torch.manual_seed(1220+rank)
            x=torch.randn(m,5120,device='cuda',dtype=torch.bfloat16)*magnitude
            residual=torch.zeros_like(x)
            gamma=torch.zeros(5120,device='cuda',dtype=torch.bfloat16)
            y=torch.empty_like(x);r=torch.empty_like(x)
            args=dict(input=x,workspace=workspace,residual_in=residual,rms_gamma=gamma,
                      rms_eps=1e-6,weight_bias=1.0,fp32_acc=True,use_oneshot=True)
            allreduce_fusion(**args,pattern=1,norm_out=y,residual_out=r)
            ref=mxfp6.quantize_mxfp8(y)
            out=torch.empty_like(ref.values).view(torch.float8_e4m3fn)
            aligned=(m+3)//4*4
            raw=torch.empty(aligned*40,device='cuda',dtype=torch.int32)
            scale=raw.as_strided((m,40),(1,aligned))
            fy=torch.empty_like(y);fr=torch.empty_like(r)
            allreduce_fusion(**args,pattern=9,norm_out=fy,residual_out=fr,
                quant_out=out,scale_out=scale,block_quant_group_size=32)
            logical=raw.view(torch.uint8).view(40,aligned,4).permute(1,0,2).reshape(aligned,160)[:m]
            expected=mxfp6.unpack_scales(ref.scales,m,5120)
            results.append(dict(rows=m,magnitude=magnitude,norm_equal=torch.equal(y,fy),
                residual_equal=torch.equal(r,fr),code_mismatches=int((out.view(torch.uint8)!=ref.values).sum()),
                scale_mismatches=int((logical!=expected).sum())))
    torch.cuda.synchronize()
    Path(output,f'rank{rank}.json').write_text(json.dumps(results,indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,args=('file://'+tmp+'/rendezvous',str(a.output.resolve())),nprocs=2,join=True)


if __name__=='__main__':main()
