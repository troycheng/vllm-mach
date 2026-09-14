# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only teacher sampling; raw logits and logprob computation stay intact.

The first forced output reinstates the last original prompt token. Every
subsequent gold token is then scored in a fixed physical-M request batch. No method
from this module is installed in a throughput service.
"""
from pathlib import Path
import hashlib
import inspect
import os
import sys

KEY='k4o_teacher_w6_all'

def install(worker):
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.distributed import get_tensor_model_parallel_rank
    physical_rows=32
    state={'physical_rows':physical_rows}
    worker._k4o_teacher_shape_state=state
    assert not hasattr(Sampler,'_k4o_teacher_originals'), 'teacher diagnostic installed twice'
    sampler_file=Path(inspect.getfile(Sampler))
    logprob_file=sampler_file.with_name('logprob.py')
    assert hashlib.sha256(sampler_file.read_bytes()).hexdigest()=='408b08c4ac66de271f476460490f360ef2a9b73528a346d132a22326e1674369'
    assert hashlib.sha256(logprob_file.read_bytes()).hexdigest()=='f47479a818192acb6b297c79fa67a3b2f9820e3492a394d5adda777e9b7c832e'
    source=inspect.getsource(Sampler.__call__)
    assert 'compute_topk_scores(' in source and 'logits = processed_logits' in source
    originals=(Sampler.add_request,Sampler.sample,Sampler.__call__)
    Sampler._k4o_teacher_originals=originals
    def add_request(self,req_idx,prompt_len,sampling_params):
        originals[0](self,req_idx,prompt_len,sampling_params)
        if not hasattr(self,'_k4o_teacher'):
            self._k4o_teacher={};self._k4o_observations=[]
        self._k4o_teacher.pop(req_idx,None)
        value=(sampling_params.extra_args or {}).get(KEY)
        if value is not None:
            assert self.logprobs_mode=='raw_logprobs'
            assert sampling_params.temperature==0.0 and sampling_params.ignore_eos
            assert sampling_params.logprobs==1
            assert prompt_len>0 and len(value['forced_ids'])==sampling_params.max_tokens
            self._k4o_teacher[req_idx]={**value,'prompt_len':prompt_len}
    def sample(self,logits,expanded_idx_mapping,idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,*,return_logprobs):
        physical_rows=state['physical_rows']
        table=getattr(self,'_k4o_teacher',{})
        active=[int(i) in table for i in idx_mapping_np]
        if not any(active):
            return originals[1](self,logits,expanded_idx_mapping,idx_mapping,idx_mapping_np,pos,input_ids,expanded_local_pos,return_logprobs=return_logprobs)
        assert all(active) and len(active)==physical_rows and logits.shape[0]==physical_rows, {'active':active,'logits_shape':tuple(logits.shape),'idx_mapping':idx_mapping_np.tolist()}
        assert return_logprobs and self.logprobs_mode=='raw_logprobs'
        assert not torch.cuda.is_current_stream_capturing()
        positions=pos.detach().cpu().tolist()
        assert len(positions)==physical_rows
        forced=[];offsets=[]
        for req_idx,position in zip(idx_mapping_np,positions,strict=True):
            info=table[int(req_idx)];offset=int(position)-info['prompt_len']+1
            assert offset>=0, 'partial prefill: all initial prompts must fit in one physical batch'
            offsets.append(offset)
            # Async serving may execute an unused tail step after max_tokens.
            # It cannot alter any scored gold prefix; retain that observation.
            forced.append(info['forced_ids'][offset] if offset<len(info['forced_ids']) else info['pad_id'])
        assert len(set(offsets))==1, 'teacher requests lost the synchronized decode barrier'
        self._k4o_last_offsets=offsets
        sampled=torch.tensor(forced,dtype=torch.int64,device=logits.device)
        # The caller subsequently computes normal raw top-k + sampled-token
        # logprobs FROM THE ORIGINAL logits. No masking or logit edit is applied.
        return sampled,logits
    def call(self,logits,input_batch):
        physical_rows=state['physical_rows']
        self._k4o_last_offsets=None
        result=originals[2](self,logits,input_batch)
        offsets=getattr(self,'_k4o_last_offsets',None)
        if offsets is not None:
            num_tokens=int(input_batch.input_ids.numel())
            assert int(input_batch.num_tokens)==num_tokens
            num_reqs=int(input_batch.num_reqs)
            if offsets[0]>0:assert num_tokens==num_reqs==physical_rows,(num_tokens,num_reqs,physical_rows)
            self._k4o_observations.append({'offset':offsets[0],'num_tokens':num_tokens,'num_reqs':num_reqs,
                'ids':[self._k4o_teacher[int(i)]['id'] for i in input_batch.idx_mapping_np]})
        return result
    Sampler.add_request=add_request;Sampler.sample=sample;Sampler.__call__=call
    runner=worker.model_runner
    # Native plan identities are checked by shape_inspector; the hook checks only the actual physical decode batch shape.
    assert os.environ['K4O_QUALITY_ARM']=='m32'
    sampler=runner.sampler
    assert isinstance(sampler,Sampler), 'requires the frozenV2 worker sampler'
    assert sampler.logprobs_mode=='raw_logprobs'
    if not hasattr(sampler,'_k4o_observations'):sampler._k4o_observations=[]
    worker._k4o_teacher_sampler=sampler
    p=Path(inspect.getfile(Sampler));hp=Path(__file__)
    return {'rank':get_tensor_model_parallel_rank(),'sampler_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'hook_sha256':hashlib.sha256(hp.read_bytes()).hexdigest(),'logprob_sha256':hashlib.sha256(logprob_file.read_bytes()).hexdigest(),'physical_rows':physical_rows,'raw_logprobs':True}

def observations(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    sampler=worker._k4o_teacher_sampler
    rows=sampler._k4o_observations
    sampler._k4o_observations=[]
    return {'rank':get_tensor_model_parallel_rank(),'calls':rows}


def set_physical_rows(worker, rows):
    from vllm.distributed import get_tensor_model_parallel_rank
    assert rows in (4,16,24,32)
    sampler=worker._k4o_teacher_sampler
    assert not sampler._k4o_observations, 'unconsumed previous-shape observations'
    worker._k4o_teacher_shape_state['physical_rows']=rows
    return {'rank':int(get_tensor_model_parallel_rank()),'physical_rows':rows,'sampler_sha256':hashlib.sha256(Path(inspect.getfile(type(sampler))).read_bytes()).hexdigest(),'hook_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'logprob_sha256':hashlib.sha256(Path(inspect.getfile(type(sampler))).with_name('logprob.py').read_bytes()).hexdigest(),'raw_logprobs':True}
