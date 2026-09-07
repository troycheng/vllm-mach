"""Opt-in, version-locked TP2 BF16 prefill communication path."""
import importlib.metadata
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)
_loaded = False
_workspace = None
_verified_norms = set()
MIN_WORKSPACE_BYTES = 84_049_920


def eligible(shape, dtype, tp_size):
    import torch
    return (os.getenv('VLLM_MACH_LOSSLESS_PREFILL', '0') == '1'
            and tp_size == 2 and tuple(shape) == (4096, 5120)
            and dtype == torch.bfloat16)


def validate_workspace(workspace, rank):
    if workspace is None or workspace.backend != 'trtllm':
        raise RuntimeError('Lossless prefill requires a trtllm workspace')
    meta = workspace.metadata
    if (meta.get('tp_size') != 2 or meta.get('tp_rank') != rank
            or meta.get('hidden_dim') != 5120
            or meta.get('max_token_num', 0) < 4096
            or meta.get('buffer_size', 0) < MIN_WORKSPACE_BYTES):
        raise RuntimeError('Lossless prefill workspace metadata/capacity mismatch')


def run(hidden_states, residual, norm, norm_out, max_token_num):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import get_fi_ar_workspace

    global _loaded, _workspace
    if torch.cuda.get_device_capability(hidden_states.device) != (12, 0):
        raise RuntimeError('Lossless prefill has only been validated on SM120')
    rank = get_tensor_model_parallel_rank()
    workspace = get_fi_ar_workspace(
        world_size=2, rank=rank, max_token_num=max_token_num,
        hidden_dim=5120, dtype=hidden_states.dtype, group=get_tp_group().cpu_group,
    )
    validate_workspace(workspace, rank)
    if not _loaded:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Initialize lossless prefill before CUDA Graph capture')
        for package, expected in [('vllm', '0.28.0'), ('flashinfer-python', '0.6.18')]:
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f'Lossless prefill requires {package}=={expected}')
        spec = importlib.util.find_spec('mach_lossless_prefill_ext')
        if spec is None or not spec.origin:
            raise RuntimeError('Build/install native/lossless_prefill before enabling this profile')
        torch.ops.load_library(spec.origin)
        _loaded, _workspace = True, workspace
        logger.warning('Mach lossless prefill active: rank=%s shape=4096x5120 buffer_size=%s',
                       rank, workspace.metadata['buffer_size'])
    if workspace is not _workspace:
        raise RuntimeError('Lossless prefill workspace changed; restart the worker')
    verify = os.getenv('VLLM_MACH_LOSSLESS_PREFILL_VERIFY', '0') == '1' and norm not in _verified_norms
    if verify:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Lossless prefill verification must run outside Graph capture')
        from vllm.compilation.passes.fusion.allreduce_rms_fusion import flashinfer_trtllm_fused_allreduce_norm
        reference_input, reference_norm = hidden_states.clone(), torch.empty_like(norm_out)
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=reference_input, residual=residual, rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon, world_size=2, weight_bias=1.0,
            launch_with_pdl=True, fp32_acc=True, max_token_num=max_token_num,
            pattern_code=1, norm_out=reference_norm,
        )
    torch.ops.mach_lossless_prefill.run(
        hidden_states, residual, norm.weight, workspace.workspace_tensor,
        hidden_states, norm_out, rank, int(workspace.metadata['buffer_size']),
        float(norm.variance_epsilon), 1.0, True, True,
    )
    if verify:
        if not (torch.equal(hidden_states.view(torch.uint16), reference_input.view(torch.uint16))
                and torch.equal(norm_out.view(torch.uint16), reference_norm.view(torch.uint16))):
            raise RuntimeError('Lossless prefill differs bitwise from installed FlashInfer')
        _verified_norms.add(norm)
        logger.warning('Mach lossless prefill verified: rank=%s norm_instances=%s', rank, len(_verified_norms))
