# SPDX-License-Identifier: Apache-2.0
"""Post-capture original-FLA graphs for the qualified native MoE FP16 worker."""
import importlib
import json
import threading
from pathlib import Path

from ._prefill_graph_dispatch import GraphDispatch


def callable_id(value):
    return [id(getattr(value, '__self__', None)), id(getattr(value, '__func__', value))]


def tensor_id(value):
    return [id(value), value.data_ptr(), list(value.shape), list(value.stride()),
            str(value.dtype), str(value.device)]


def _retain_kernel_run(leaf):
    """Retain the original bound method, as the qualified capture restore did.

    JITFunction.run is a descriptor: two reads can create distinct method
    objects. Keep one instance binding for the existing strict owner guards.
    """
    original = leaf.run
    leaf.run = original
    return original


class Binding:
    def __init__(self, worker):
        import torch
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
        from vllm.model_executor.models import qwen3_5
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
        from vllm.third_party.flash_linear_attention.ops import chunk, index
        from vllm_mach.mxfp6.gdn import legacy
        self.worker, self.torch, self.chunk = worker, torch, chunk
        self.runner = worker.model_runner
        # legacy.install already validates these installed FLA source gates.
        # Reuse that normal qualification instead of shipping experiment files.
        self.source = json.loads((Path(legacy.__file__).parents[1]
                                  / 'profile/legacy_sources.json').read_text())
        cfg = worker.vllm_config
        assert cfg.parallel_config.tensor_parallel_size == 2
        assert cfg.scheduler_config.max_num_seqs == 64
        assert cfg.cache_config.kv_cache_memory_bytes == 8589934592
        assert cfg.cache_config.mamba_ssm_cache_dtype == 'float16'
        assert cfg.speculative_config is None and not cfg.parallel_config.enable_dbo
        outer = self.runner.model
        assert type(outer) is qwen3_5.Qwen3_5MoeForConditionalGeneration
        language = outer.language_model
        assert type(language) is qwen3_5.Qwen3_5MoeForCausalLM
        self.backbone = language.model
        assert isinstance(self.backbone, qwen3_5.Qwen3_5Model)
        self.layers = [layer for layer in self.backbone.modules()
                       if type(layer) is qwen.QwenGatedDeltaNetAttention]
        assert len(self.layers) == 30
        assert {id(layer) for layer in self.layers} == {
            id(layer) for layer in outer.modules() if type(layer) is qwen.QwenGatedDeltaNetAttention}
        self.methods = []
        for layer in self.layers:
            obj = layer.chunk_gated_delta_rule
            assert type(obj) is qwen.ChunkGatedDeltaRule
            assert obj.gdn_prefill_backend == 'triton' and layer.gdn_prefill_backend == 'triton'
            original = obj._forward_method
            assert getattr(original, '__self__', None) is obj
            assert getattr(original, '__func__', None) is qwen.ChunkGatedDeltaRule.forward_native
            assert layer.kv_cache[1].dtype == torch.float16 and layer._mach_native_mixed_fp16
            self.methods.append((layer, obj, original))
        assert len({id(obj) for _, obj, _ in self.methods}) == 30
        manager = self.runner.cudagraph_manager
        assert manager is not None and manager._graphs_captured and manager.graphs
        assert legacy.enabled(), 'Actual native worker must select the original qualified FLA policy'
        self.stream = self.runner.main_stream
        self.device = self.layers[0].kv_cache[1].device
        assert self.stream.device == self.device and torch.cuda.current_device() == self.device.index
        self.thread = threading.get_ident()
        self.expected = self.identity()
        choices = json.loads((Path(legacy.__file__).parents[1] / 'profile/fla_choices.json').read_text())['kernels']
        self.installed = []
        for row in choices:
            leaf = getattr(importlib.import_module(row['module']), row['function'])
            tuner = None
            while not isinstance(leaf, JITFunction):
                if isinstance(leaf, Autotuner):
                    tuner = leaf
                leaf = leaf.fn
            assert tuner is not None
            self.installed.append((row, leaf, _retain_kernel_run(leaf), tuner))
        assert len(self.installed) == 6
        self.graph = GraphDispatch(chunk, self.installed, self.stream)
        self.graph.capture_stream = torch.cuda.Stream(device=self.device)
        self.memory = {}
        self.prepared = []
        self.original = self.graph.original
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.stream.synchronize()
            self.memory['before'] = self.memory_info()
            # Same four real (N, NT) banks; only token capacity is padded.
            # No KV state or model input is borrowed for startup capture.
            for lengths in ((2034,), (862, 1173), (1827,), (1931,)):
                T, N = sum(lengths), len(lengths)
                cumulative = [0]
                for length in lengths:
                    cumulative.append(cumulative[-1] + length)
                cpu = torch.tensor(cumulative, dtype=torch.int32)
                cu = cpu.to(self.device)
                ci = index.prepare_chunk_indices(cpu, 64).to(self.device)
                co = index.prepare_chunk_offsets(cpu, 64).to(self.device)
                q = torch.full((1, T, 8, 128), .0625, dtype=torch.bfloat16, device=self.device)
                k = torch.full_like(q, .0625)
                v = torch.full((1, T, 16, 128), .01, dtype=torch.bfloat16, device=self.device)
                g = torch.full((1, T, 16), -.02, dtype=torch.float32, device=self.device)
                beta = torch.full_like(g, .25)
                initial = torch.zeros((N, 16, 128, 128), dtype=torch.float16, device=self.device)
                output = torch.empty((T, 16, 128), dtype=torch.bfloat16, device=self.device)
                args = dict(q=q, k=k, v=v, g=g, beta=beta, scale=128**-.5,
                            initial_state=initial, output_final_state=True, cu_seqlens=cu,
                            chunk_indices=ci, chunk_offsets=co, core_attn_out=output)
                prepared = self.graph.prepare(args)
                self.prepared.append(prepared)
            del args, q, k, v, g, beta, initial, output, cu, ci, co
            self.stream.synchronize()
            self.graph.seal()
            self.memory['after'] = self.memory_info()
        self.check()

    def memory_info(self):
        torch = self.torch
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(allocated=torch.cuda.memory_allocated(self.device),
                    reserved=torch.cuda.memory_reserved(self.device), free=free, total=total)

    def identity(self):
        manager = self.runner.cudagraph_manager
        return dict(model=id(self.runner.model), language=id(self.runner.model.language_model),
                    backbone=id(self.backbone), main_stream=self.runner.main_stream.cuda_stream,
                    graph_manager=id(manager), captured=manager._graphs_captured,
                    graphs=sorted([repr(key), id(value)] for key, value in manager.graphs.items()),
                    kv_bytes=self.worker.vllm_config.cache_config.kv_cache_memory_bytes,
                    capacity=self.worker.vllm_config.scheduler_config.max_num_seqs,
                    layers=[dict(layer=id(layer), prefix=layer.prefix,
                                 chunk=id(obj), native_method=callable_id(original),
                                 state=tensor_id(layer.kv_cache[1]), backend=obj.gdn_prefill_backend)
                            for layer, obj, original in self.methods])

    def check(self):
        if self.identity() != self.expected:
            raise RuntimeError('Actual model/30-layer/state/stream/FULL-graph ownership changed')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Control RPC must be outside original capture')

    def methods_original(self):
        return all(callable_id(obj._forward_method) == callable_id(original)
                   for _, obj, original in self.methods)


_STATE = '_mach_prefill_graph'


def _selected(worker):
    # Share the already-qualified cold native model/capture selection with head.
    from ..head_scale import _selected as native_selected
    if not native_selected(worker):
        return False
    config = worker.vllm_config
    if (config.scheduler_config.max_num_seqs != 64
            or config.cache_config.kv_cache_memory_bytes != 8589934592
            or config.speculative_config is not None
            or config.parallel_config.enable_dbo):
        return False
    from . import legacy
    return legacy.enabled()


def install(worker):
    """Prepare four private banks once, after the ordinary FULL captures."""
    if not _selected(worker):
        return False
    if hasattr(worker, _STATE):
        raise RuntimeError('Native prefill graphs are already installed')
    binding = Binding(worker)
    binding.graph.enable()
    setattr(worker, _STATE, binding)
    return True


def inspect_worker(worker):
    """Cold receipt only; scored execution has no observers or counters."""
    binding = getattr(worker, _STATE, None)
    if binding is None:
        return dict(installed=False)
    return dict(installed=binding.chunk.chunk_gated_delta_rule_fwd is binding.graph.run,
                banks=[list(key) for key in binding.graph.banks], token_capacity=2048,
                layers=len(binding.methods), memory=binding.memory,
                kv_bytes=binding.expected['kv_bytes'], capacity=binding.expected['capacity'],
                main_stream=binding.stream.cuda_stream,
                capture_stream=binding.graph.capture_stream.cuda_stream)
