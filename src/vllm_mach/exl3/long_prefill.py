"""Opt-in exact-shape prefill collective; M4096 keeps its separate implementation."""
import importlib.metadata
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)
ROWS = frozenset((1024, 1052, 3000, 3001, 3002, 3003, 3012, 3013, 3014, 3015,
                  3020, 3021, 3022, 3023, 3028, 3029, 3030, 3031, 3093, 3094,
                  3095, 3185, 3244, 3658, 3842, 3845, 3866, 3869, 3890, 3891, 3893, 3895))
_workspace = None
_active = set()
_verified = set()


def eligible(shape, dtype, tp_size):
    import torch
    return (os.getenv('VLLM_MACH_LOSSLESS_PREFILL_LONG', '0') == '1'
            and tp_size == 2 and len(shape) == 2 and shape[1] == 5120 and shape[0] in ROWS
            and dtype == torch.bfloat16 and not torch.cuda.is_current_stream_capturing())


def workspace_bytes(rows):
    return rows * 5120 * 4 + (rows * 5120 // 256) * 4


def validate_workspace(workspace, rank, rows):
    if rows not in ROWS or workspace is None or workspace.backend != 'trtllm':
        raise RuntimeError('Long prefill requires a supported shape and trtllm workspace')
    meta = workspace.metadata
    if (meta.get('tp_size') != 2 or meta.get('tp_rank') != rank
            or meta.get('hidden_dim') != 5120 or meta.get('max_token_num', 0) < rows
            or meta.get('buffer_size', 0) < workspace_bytes(rows)):
        raise RuntimeError('Long prefill workspace metadata/capacity mismatch')


def run(hidden_states, residual, norm, norm_out, max_token_num):
    import torch
    import flashinfer.comm
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import get_fi_ar_workspace
    global _workspace
    rows = hidden_states.shape[0]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Long prefill dispatch is eager-only')
    if torch.cuda.get_device_capability(hidden_states.device) != (12, 0):
        raise RuntimeError('Long prefill has only been validated on SM120')
    rank = get_tensor_model_parallel_rank()
    workspace = get_fi_ar_workspace(world_size=2, rank=rank, max_token_num=max_token_num,
                                    hidden_dim=5120, dtype=hidden_states.dtype, group=get_tp_group().cpu_group)
    validate_workspace(workspace, rank, rows)
    if _workspace is None:
        for package, expected in [('vllm', '0.28.0'), ('flashinfer-python', '0.6.18')]:
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f'Long prefill requires {package}=={expected}')
        spec = importlib.util.find_spec('mach_lossless_prefill_long_ext')
        if spec is None or not spec.origin:
            raise RuntimeError('Build/install native/lossless_prefill before enabling long prefill')
        torch.ops.load_library(spec.origin)
        _workspace = workspace
    if workspace is not _workspace:
        raise RuntimeError('Long prefill workspace changed; restart workers')
    if rows not in _active:
        logger.warning('Mach long prefill active: rank=%s M=%s', rank, rows)
        _active.add(rows)
    key = (rows, norm)
    verify = os.getenv('VLLM_MACH_LOSSLESS_PREFILL_VERIFY', '0') == '1' and key not in _verified
    if verify:
        reference, reference_norm = hidden_states.clone(), torch.empty_like(norm_out)
        flashinfer.comm.allreduce_fusion(reference, workspace=workspace, pattern=1,
            launch_with_pdl=True, trigger_completion_at_end=True, output=None,
            residual_out=reference, norm_out=reference_norm, residual_in=residual,
            rms_gamma=norm.weight, rms_eps=norm.variance_epsilon, use_oneshot=rows <= 3276,
            fp32_acc=True, weight_bias=1.0)
    torch.ops.mach_lossless_prefill_long.run(hidden_states, residual, norm.weight,
        workspace.workspace_tensor, hidden_states, norm_out, rank,
        int(workspace.metadata['buffer_size']), float(norm.variance_epsilon), 1.0, True)
    if verify:
        if not (torch.equal(hidden_states.view(torch.int16), reference.view(torch.int16))
                and torch.equal(norm_out.view(torch.int16), reference_norm.view(torch.int16))):
            raise RuntimeError('Long prefill differs bitwise from installed FlashInfer')
        _verified.add(key)
        count = sum(m == rows for m, _ in _verified)
        if count == 128:
            logger.warning('Mach long prefill verified: rank=%s M=%s norm_instances=128', rank, rows)
