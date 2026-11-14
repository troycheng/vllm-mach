# SPDX-License-Identifier: Apache-2.0
"""Replay the warmed M4/K128 selector on the current adjusted coarse logits."""
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from types import MethodType
import hashlib

_STATE = '_mach_2b_selector_graph'


class SelectorGraph:
    def __init__(self, state):
        import torch
        from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import _HybridHeadGraphs
        self.torch, self.state = torch, state
        self.original = state.select_candidates
        self.signature_fn = _HybridHeadGraphs.signature
        self.signature = self.signature_fn(state)
        self.head_graphs = state._head_graphs
        self.stream = torch.cuda.current_stream(state.weight.device)
        if self.stream != state._head_graphs.stream:
            raise RuntimeError('Selector and warmed head streams differ')
        self.lock = Lock()
        self.replays = self.fallbacks = 0
        self.input = torch.zeros((4, state.output_size), dtype=torch.bfloat16,
                                 device=state.weight.device)
        self.original(self.input, top_k=20)
        self.stream.synchronize()  # Preparation only, before serving requests.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.indices = self.original(self.input, top_k=20)

    def select(self, coarse_logits, *, top_k=None):
        torch, state = self.torch, self.state
        width = state.candidates if top_k is None else state.candidate_count_for_topk(top_k)
        if (tuple(coarse_logits.shape) != (4, 248320) or width != 128
                or self.signature_fn(state) != self.signature
                or state._head_graphs is not self.head_graphs
                or not torch.is_inference_mode_enabled()
                or coarse_logits.dtype != torch.bfloat16
                or coarse_logits.device != state.weight.device
                or not coarse_logits.is_contiguous()
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(coarse_logits.device) != self.stream
                or not self.lock.acquire(blocking=False)):
            self.fallbacks += 1
            return self.original(coarse_logits, top_k=top_k)
        try:
            # Presence penalties already ran on this stream. Never retain or
            # reuse the previous call's input or return graph-owned indices.
            self.input.copy_(coarse_logits)
            self.graph.replay()
            output = self.indices.clone()
            self.replays += 1
            return output
        finally:
            self.lock.release()


def install(worker, state):
    contract = getattr(worker, '_mach_2b_contract', {})
    if (contract.get('mode') != 'production'
            or contract.get('ssm_dtype') != 'float32'
            or not contract.get('hybrid_greedy_head')
            or state._head_graphs is None):
        return False
    if getattr(state, _STATE, None) is not None:
        return False
    if version('vllm').split('+', 1)[0] != '0.33.0':
        raise RuntimeError('2B selector graph requires the selected vLLM runtime')
    import torch
    from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import HybridNvfp4LmHead
    if (type(state) is not HybridNvfp4LmHead or state.backend != 'b12x'
            or (state.input_size, state.output_size, state.candidates) != (2048, 248320, 128)
            or 'select_candidates' in vars(state)
            or torch.cuda.is_current_stream_capturing()):
        return False
    with torch.inference_mode():
        graph = SelectorGraph(state)

    @wraps(HybridNvfp4LmHead.select_candidates)
    def select(self, coarse_logits, *, top_k=None):
        return graph.select(coarse_logits, top_k=top_k)

    state.select_candidates = MethodType(select, state)
    setattr(state, _STATE, graph)
    return True


def receipt(state):
    graph = getattr(state, _STATE, None)
    if graph is None:
        return {'installed': False}
    return {'installed': True, 'rows': 4, 'candidate_width': 128,
            'replays': graph.replays, 'fallbacks': graph.fallbacks,
            'owned_outputs': True, 'input_bytes': graph.input.numel() * graph.input.element_size(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
