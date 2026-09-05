# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense EXL3 provider for the pinned vLLM v0.28 surface.

The initial compatibility gate uses a Qwen3.8-27B K5/K6 checkpoint. Shape-
specific fast paths remain fail-closed outside that validated layout.
"""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import importlib
import os
import re
import sys
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter
from transformers import PretrainedConfig

from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from . import mxfp6_hybrid
from .hadamard import hadamard_fold_weight_chunked

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_MCG_SENTINEL = 0xCBAC1FED
_MUL1_SENTINEL = 0x83DCD12D
_HADAMARD_BLOCK = 128
_EXL3_EXT: Any | None = None
_B12X_TRELLIS_LINEAR_API: Any | None = None
_QKV_MGEMM_ENABLED = os.environ.get("EXL3_QKV_MGEMM", "0") == "1"
_BF16_IO_ENABLED = os.environ.get("EXL3_BF16_IO", "0") == "1"
_BF16_IO_M24_ENABLED = os.environ.get("EXL3_BF16_IO_M24", "0") == "1"
_BF16_IO_M32_ENABLED = os.environ.get("EXL3_BF16_IO_M32", "0") == "1"
_BF16_IO_TILE_M32_ENABLED = os.environ.get("EXL3_BF16_IO_TILE_M32", "0") == "1"
_EXL3_GEMM_PRIMED_SIGNATURES: set[tuple[int, int, int, int, int, int]] = set()
_EXL3_QKV_MGEMM_PRIMED_SIGNATURES: set[tuple[int, int, int, int, int, int, int]] = set()
_B12X_TRELLIS_WARMED_DEVICES: set[tuple[int, int]] = set()
_B12X_SELFTEST = os.environ.get("VLLM_EXL3_B12X_SELFTEST", "0") == "1"
_B12X_SELFTEST_DONE: set[tuple] = set()
_RECON_ARENA: dict[int, torch.Tensor] = {}
_PREFILL_RECONSTRUCT_CACHE = (
    os.environ.get("VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE", "1") == "1"
)
_PREFILL_FUSED_RECONSTRUCT_MIN_M = int(
    os.environ.get("VLLM_EXL3_PREFILL_FUSED_RECONSTRUCT_MIN_M", "0")
)
_B12X_MIN_M = int(os.environ.get("VLLM_EXL3_B12X_MIN_M", "0"))
_B12X_TRELLIS_C_TMP_CAP = 192 * 4 * 64 * 256
_B12X_C_TMP_SHARED: dict[int, torch.Tensor] = {}
_B12X_TRELLIS_BUF_CACHE_MAX_ROWS = 128
_EMBED_ONLINE_EPS = 1.0e-8
_SUPPORTED_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
_SUPPORTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_SUPPORTED_HIDDEN_SIZE = 5120

ShardId = str | int | tuple[int, ...] | None

def _embed_online_bits() -> int | None:
    """Return the online embedding-quantization width, or None when disabled.

    ``VLLM_EXL3_EMBED_ONLINE_BITS`` selects per-row online quantization of the
    token embedding table (``VocabParallelEmbedding``) at load time. Accepted
    values: unset/0 = off; an integer from 3 to 8. Only ``8`` (int8) and ``6``
    (packed int6) reduce the table footprint; other widths quantize to the
    requested precision but remain stored in an int8 container.
    """
    raw = os.environ.get("VLLM_EXL3_EMBED_ONLINE_BITS")
    if raw is None or not raw.strip():
        return None
    try:
        bits = int(raw)
    except ValueError:
        raise ValueError(
            f"VLLM_EXL3_EMBED_ONLINE_BITS must be an integer from 3 to 8, "
            f"got {raw!r}"
        ) from None
    if bits not in range(3, 9):
        raise ValueError(
            f"VLLM_EXL3_EMBED_ONLINE_BITS must be from 3 to 8, got {bits}"
        )
    return bits

class Exl3OnlineEmbeddingMethod(QuantizeMethodBase):
    """Env-gated online quantization for ``VocabParallelEmbedding`` token tables.

    The checkpoint is loaded as BF16 (so the stock vocab-parallel weight loader
    works unchanged) and converted in ``process_weights_after_loading`` to a
    compact per-row format, freeing the BF16 tensor:

    * ``bits == 8``: per-row symmetric int8. ``q`` is int8 ``[V, H]`` and the
      per-row scale is fp16 ``[V]``. Footprint ~1.27 GB (decimal) for 248320x5120.
    * ``bits == 6``: per-row symmetric int6, packed four elements to three
      bytes (4*6 = 24 bits). ``q`` is uint8 ``[V, 3H/4]`` and the scale is
      fp16 ``[V]``. Footprint ~0.95 GB (decimal) for 248320x5120 (requires ``H % 4``).
    * other ``bits`` in ``3..7``: quantized to the requested precision but kept
      in an int8 container (no extra footprint reduction vs ``bits == 8``).

    ``embedding()`` performs a CUDA-graph-safe gather + dequant: it indexes the
    compact weight by token id (a pure gather, no host sync, no
    ``.item()``), casts to bf16, and multiplies by the gathered per-row scale.
    Steady-state allocations are limited to the gathered rows.

    EXL3 Trellis K6/K8 would be preferable (smaller, KLD-safe) but the shipped
    exllamav3 extension only exposes ``reconstruct`` / ``reconstruct_slice``
    over *contiguous, 128-aligned* bands of the matrix N dimension
    (``reconstruct.cu`` lines 118-121), not an arbitrary-row indexed gather.
    An embedding lookup needs scattered vocab rows, so a Trellis-backed gather
    would either reconstruct the whole table (defeating the savings) or launch
    one ``reconstruct_slice`` per 128-row band touched by the batch (dynamic
    count, not graph-safe). See ``upstream/embed-online-quant/issue-body.md``.
    """

    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = int(bits)
        # Only 6 packs to a sub-byte container; 8 uses native int8. Every other
        # width quantizes to N-bit precision but stays in an int8 container.
        self.packed: bool = self.bits == 6

    # -- QuantizeMethodBase -------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create the BF16 loading weight, mirroring UnquantizedEmbeddingMethod.

        The quantized tensors are materialized only in
        ``process_weights_after_loading``; until then the layer carries a normal
        BF16 ``weight`` so the vocab-parallel weight loader is unaffected.
        """
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Encode the loaded BF16 table to the compact per-row format and free it."""
        w = layer.weight.data
        prefix = getattr(layer, "prefix", type(layer).__name__)
        device = w.device
        num_rows, hidden = w.shape

        if self.packed and hidden % 4 != 0:
            raise ValueError(
                f"VLLM_EXL3_EMBED_ONLINE_BITS=6 requires hidden_dim divisible by "
                f"4, got {hidden} for {prefix}"
            )

        if not self.packed and self.bits != 8:
            logger.warning_once(
                "VLLM_EXL3_EMBED_ONLINE_BITS=%d for %s is stored in an int8 "
                "container; only bits=6 packs to a sub-byte footprint.",
                self.bits, prefix,
            )

        # Per-row symmetric scale. Compute in fp32 to avoid bf16 rounding in
        # the division; the stored scale stays fp16 (negligible size).
        # Chunked: a full-table fp32 copy is 4.74 GiB for 248320x5120 and
        # OOMs when quantized weights are large (measured under all-FP6).
        max_q = (1 << (self.bits - 1)) - 1  # 127 for 8, 31 for 6, etc.
        amax = torch.empty(num_rows, dtype=torch.float32, device=device)
        _AMAX_CHUNK = 16384
        for r0 in range(0, num_rows, _AMAX_CHUNK):
            r1 = min(r0 + _AMAX_CHUNK, num_rows)
            amax[r0:r1] = w[r0:r1].to(torch.float32).abs().amax(dim=1)
        scale = amax.clamp(min=_EMBED_ONLINE_EPS) / max_q
        scale_fp16 = scale.to(torch.float16)
        del amax

        # Chunk the encode over vocab rows: full-table fp32/int32 transients
        # for a 248320x5120 table are ~4.7+1.3 GiB and OOM the loader at peak
        # (measured: 1.19 GiB `val` alloc failed with model fully resident).
        # 16384-row chunks cap transients at ~0.5 GiB.
        _CHUNK = 16384
        if self.packed:
            q_weight = torch.empty(
                num_rows, (hidden // 4) * 3, dtype=torch.uint8, device=device
            )
        else:
            q_weight = torch.empty(
                num_rows, hidden, dtype=torch.int8, device=device
            )
        q_lo = -(1 << (self.bits - 1))
        for r0 in range(0, num_rows, _CHUNK):
            r1 = min(r0 + _CHUNK, num_rows)
            w_c = w[r0:r1].to(torch.float32)
            q_fp32 = (w_c / scale[r0:r1].unsqueeze(1)).round()
            del w_c
            if self.packed:
                # int6: signed [-32, 31] -> unsigned [0, 63] for packing.
                u = (q_fp32.clamp(-32, 31) + 32).to(torch.uint8)
                del q_fp32
                u = u.reshape(r1 - r0, hidden // 4, 4).to(torch.int32)
                val = (
                    u[..., 0]
                    | (u[..., 1] << 6)
                    | (u[..., 2] << 12)
                    | (u[..., 3] << 18)
                )  # int32 [chunk, H/4]
                del u
                b0 = (val & 0xFF).to(torch.uint8)
                b1 = ((val >> 8) & 0xFF).to(torch.uint8)
                b2 = ((val >> 16) & 0xFF).to(torch.uint8)
                del val
                q_weight[r0:r1] = (
                    torch.stack((b0, b1, b2), dim=-1)
                    .reshape(r1 - r0, (hidden // 4) * 3)
                )
                del b0, b1, b2
            else:
                # int8 container (native for bits==8, N-bit range otherwise).
                q_weight[r0:r1] = q_fp32.clamp(q_lo, max_q).to(torch.int8)
                del q_fp32
        del scale
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # Free the BF16 table before registering the compact tensors.
        _mem_before = (
            torch.cuda.memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        )
        del layer.weight
        del w
        layer.register_buffer("q_weight", q_weight, persistent=False)
        layer.register_buffer("embed_scale", scale_fp16, persistent=False)
        # 0-row stub keeps `layer.weight` addressable for the MTP embed-sharing
        # pre-check (llm_base_proposer.py:1573 isinstance + .shape[-1]); both
        # sharing paths then share the whole MODULE, inheriting q_weight.
        layer.register_parameter(
            "weight",
            Parameter(
                torch.empty(0, hidden, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            ),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _mem_after = (
            torch.cuda.memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        )
        logger.info(
            "EXL3 embed online K%d conversion complete for %s %.2f→%.2f GiB "
            "(Δ%.2f)",
            self.bits, prefix, _mem_before, _mem_after, _mem_after - _mem_before,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Exl3OnlineEmbeddingMethod only supports embedding() gather; "
            "apply() is not used by VocabParallelEmbedding."
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """CUDA-graph-safe gather + dequant of the compact embedding table."""
        # Direct advanced indexing (a pure gather) is used instead of
        # F.embedding so integer-dtype / 1-D tensors are accepted uniformly;
        # it is CUDA-graph-capturable and never host-syncs.
        scale = layer.embed_scale[input_].to(torch.bfloat16).unsqueeze(-1)
        if self.packed:
            packed_cols = layer.q_weight.shape[1]
            hidden = (packed_cols // 3) * 4
            packed = layer.q_weight[input_]  # uint8 [N, 3H/4]
            n = packed.shape[0]
            packed = packed.reshape(n, hidden // 4, 3).to(torch.int32)
            val = (
                packed[..., 0]
                | (packed[..., 1] << 8)
                | (packed[..., 2] << 16)
            )  # [N, H/4]
            u = torch.stack(
                (val & 0x3F, (val >> 6) & 0x3F, (val >> 12) & 0x3F,
                 (val >> 18) & 0x3F),
                dim=-1,
            ).reshape(n, hidden)  # int32 [N, H]
            q = (u - 32).to(torch.bfloat16)
        else:
            q = layer.q_weight[input_].to(torch.bfloat16)
        return q * scale

    def tie_weights(
        self, layer: torch.nn.Module, embed_tokens: torch.nn.Module
    ) -> torch.nn.Module:
        raise NotImplementedError(
            "Online embedding quantization (VLLM_EXL3_EMBED_ONLINE_BITS) is "
            "incompatible with tied word embeddings; the EXL3 stack already "
            "unties lm_head (see tie_word_embeddings override)."
        )

def _install_embed_online_hook() -> None:
    """Install Exl3OnlineEmbeddingMethod on VocabParallelEmbedding at init time.

    Model code constructs ``embed_tokens = VocabParallelEmbedding(vocab, hidden)``
    WITHOUT passing quant_config (qwen3_5.py:243-246), so
    ``Exl3Config.get_quant_method`` is never consulted for the token table.
    This wraps ``__init__`` to swap the quant method after construction; the
    BF16 weight created by UnquantizedEmbeddingMethod.create_weights is
    byte-identical to ours, so the loader path is unchanged.
    """
    if _embed_online_bits() is None:
        return
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )
    if getattr(VocabParallelEmbedding, "_exl3_embed_online_hooked", False):
        return
    _orig_init = VocabParallelEmbedding.__init__

    def _hooked_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if type(self) is VocabParallelEmbedding and isinstance(
            self.quant_method, UnquantizedEmbeddingMethod
        ):
            bits = _embed_online_bits()
            if bits is not None:
                self.quant_method = Exl3OnlineEmbeddingMethod(bits)

    VocabParallelEmbedding.__init__ = _hooked_init
    VocabParallelEmbedding._exl3_embed_online_hooked = True
    logger.info_once(
        "EXL3 embed online hook installed (VLLM_EXL3_EMBED_ONLINE_BITS=%d)",
        _embed_online_bits(),
    )

_install_embed_online_hook()

def get_tensor_model_parallel_rank() -> int:
    """Read TP rank only after a worker constructs a quantized layer.

    v0.28 plugin discovery runs in a controller process too; importing the
    distributed singleton there is unnecessary and can bind worker lifecycle
    state before model construction. The call contract is unchanged.
    """
    from vllm.distributed import get_tensor_model_parallel_rank as get_rank

    return get_rank()

def get_tensor_model_parallel_world_size() -> int:
    """Read TP world size only after a worker constructs a quantized layer."""
    from vllm.distributed import get_tensor_model_parallel_world_size as get_size

    return get_size()

def _load_exl3_ext() -> Any:
    """Load the existing ExLlamaV3 extension only from an actual CUDA call."""

    global _EXL3_EXT
    if _EXL3_EXT is not None:
        return _EXL3_EXT

    shim = os.environ.get("VLLM_EXL3_ABI_SHIM")
    if shim:
        ctypes.CDLL(shim, mode=ctypes.RTLD_GLOBAL)

    ext_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if ext_path:
        search_dir = ext_path if os.path.isdir(ext_path) else os.path.dirname(ext_path)
        if search_dir and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        ext = importlib.import_module("exllamav3_ext")
    except Exception as exc:
        hint = (
            "Set VLLM_EXL3_EXT_PATH to the directory containing "
            "exllamav3_ext*.so (and VLLM_EXL3_ABI_SHIM when the local "
            "PyTorch ABI shim is required)."
        )
        raise RuntimeError(f"Unable to import exllamav3_ext. {hint}") from exc

    if not hasattr(ext, "exl3_gemm"):
        raise RuntimeError(
            "The imported exllamav3_ext does not export exl3_gemm; rebuild the "
            "track_a_retile extension used by this overlay."
        )
    _EXL3_EXT = ext
    return ext

def _load_b12x_trellis_linear() -> Any:
    """Resolve the native dense Trellis API lazily."""

    global _B12X_TRELLIS_LINEAR_API
    if _B12X_TRELLIS_LINEAR_API is not None:
        return _B12X_TRELLIS_LINEAR_API
    try:
        from b12x.gemm import trellis_linear
    except Exception as exc:
        raise RuntimeError(
            "Online EXL3 prefill requires b12x.gemm.trellis_linear. "
            "Install a matching B12X build."
        ) from exc
    _B12X_TRELLIS_LINEAR_API = trellis_linear
    return trellis_linear

@torch.library.custom_op(
    "vllm::exl3_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Opaque torch op around the bit-faithful ExLlamaV3 dense call.

    Prefill uses fused Hadamard reconstruction when the extension provides it,
    and otherwise retains the original reconstruct-and-fold implementation.
    Decode uses the original Trellis kernel.
    """
    ext = _load_exl3_ext()
    m = x.shape[0]
    n = trellis.shape[1] * 16
    k = x.shape[1]

    # Reconstruct+hgemm needs a full FP16 copy of the weight live at once.  For
    # the lm_head that is 5120 x 248320 x 2 = 2.37 GiB, which OOMs any profile
    # without spare headroom (the fidelity profile has ~0 after B12X prep).
    # Bound the route by reconstructed size; the default is high enough to keep
    # existing profiles on their measured path, and a profile that is tight on
    # memory opts into a smaller cap.
    _recon_max_mb = int(
        os.environ.get("VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB", "4096")
    )
    _recon_mb = (int(trellis.shape[0]) * 16) * n * 2 / 1024 / 1024
    if (
        m >= 128
        and not os.environ.get("VLLM_EXL3_PREFILL_RECONSTRUCT_M", "1") == "0"
        and _recon_mb <= _recon_max_mb
    ):
        # Reconstruct trellis codes to FP16 weight matrix (K, N)
        # Cache large weights (gate_up, >150 MB) to skip reconstruct on repeat calls
        t_k = int(trellis.shape[0]) * 16
        t_n = int(trellis.shape[1]) * 16
        weight_size_mb = t_k * t_n * 2 / 1024 / 1024  # FP16
        cache_key = trellis.data_ptr()
        if not hasattr(_exl3_gemm, '_gate_cache'):
            _exl3_gemm._gate_cache = {}  # type: ignore[attr-defined]
        _gate_cache = _exl3_gemm._gate_cache  # type: ignore[attr-defined]
        weight = None
        if weight_size_mb > 150:
            weight = _gate_cache.get(cache_key)
        # NOTE: an earlier comment here claimed this skipped caching during
        # vLLM's profiling forward (determine_available_memory).  No such check
        # existed.  The cache is therefore populated *during* profiling, so the
        # profiler both reads an inflated peak and cannot count the cached bytes
        # as free -- on a memory-tight profile that ends in "No available memory
        # for the cache blocks" with zero KV.  Caching is now explicitly
        # controllable; the default preserves the measured behaviour of the
        # profiles that have headroom for it.
        if weight is None:
            # Arena (PR #397's idea, minimal form): the reconstruct route's
            # only uncached users share one persistent fp16 buffer per device
            # instead of paying a fresh full-size torch.empty per call - on the
            # fidelity profile that was 128 x ~340 MB alloc/free per prefill
            # chunk.  Safe because ext.reconstruct overwrites every element of
            # the view (dense fill), prefill is never CUDA-graph captured, and
            # a cached weight (below) still gets its own real tensor so the
            # cache never aliases the arena.
            will_cache = weight_size_mb > 150 and _PREFILL_RECONSTRUCT_CACHE
            if will_cache:
                weight = torch.empty(
                    t_k, t_n, dtype=torch.float16, device=x.device
                )
            else:
                dev = -1 if x.device.index is None else int(x.device.index)
                arena = _RECON_ARENA.get(dev)
                if arena is None or arena.numel() < t_k * t_n:
                    arena = torch.empty(
                        t_k * t_n, dtype=torch.float16, device=x.device
                    )
                    _RECON_ARENA[dev] = arena
                weight = arena[: t_k * t_n].view(t_k, t_n)
            trellis_k = int(trellis.shape[2]) // 16
            use_fused_reconstruct = (
                _PREFILL_FUSED_RECONSTRUCT_MIN_M > 0
                and m >= _PREFILL_FUSED_RECONSTRUCT_MIN_M
                and hasattr(ext, "reconstruct_had_slice")
            )
            if use_fused_reconstruct:
                ext.reconstruct_had_slice(
                    weight,
                    trellis,
                    suh,
                    svh,
                    trellis_k,
                    mcg,
                    mul1,
                    0,
                )
                logger.warning_once(
                    "EXL3 fused prefill reconstruction active at M >= %d",
                    _PREFILL_FUSED_RECONSTRUCT_MIN_M,
                )
            else:
                ext.reconstruct(weight, trellis, trellis_k, mcg, mul1)
                weight = hadamard_fold_weight_chunked(weight, suh, svh)
            # Persistent cache: always cache large weights (no file flag needed).
            # Memory: 64 gate_up × 272 MB = 17.4 GB. With trellis weights
            # (~14 GB) and 16k KV cache (~0.7 GB), total ~32 GB. Use 16k
            # context and gpu_util=0.92 to fit.
            # Persistent cache: cache any large weight if memory allows.
            # Check free memory to avoid OOM (limits cache to available space).
            if weight_size_mb > 150 and _PREFILL_RECONSTRUCT_CACHE:
                free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
                if free_mb > weight_size_mb + 500:  # Leave 500 MB headroom
                    _gate_cache[cache_key] = weight
        output = torch.empty(m, n, dtype=torch.float16, device=x.device)
        ext.hgemm(x, weight, output)
        return output
    # Decode path: original trellis kernel
    output = torch.empty(
        (m, n),
        dtype=torch.float16,
        device=x.device,
    )
    x_had = torch.empty_like(x)
    ext.exl3_gemm(
        x,
        trellis,
        output,
        suh,
        x_had,
        svh,
        -1,
        mcg,
        mul1,
        0,
    )
    return output

@_exl3_gemm.register_fake
def _exl3_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )

@torch.library.custom_op(
    "vllm::exl3_qkv_mgemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_qkv_mgemm(
    x: torch.Tensor,
    trellis_ptrs: torch.Tensor,
    suh_ptrs: torch.Tensor,
    svh_ptrs: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
    output_size: int,
) -> torch.Tensor:
    """Run one homogeneous EXL3 projection bundle.

    The persistent pointer tensors refer to load-time, output-sliced copies of
    the original trellis weights.  Keeping this call opaque makes the existing
    extension's graph-safe multi-matrix launch visible to vLLM as one op.
    """
    ext = _load_exl3_ext()
    count = trellis_ptrs.numel()
    m, k = x.shape
    output = torch.empty(
        (count, m, output_size), dtype=torch.float16, device=x.device
    )
    x_had = torch.empty((count, m, k), dtype=torch.float16, device=x.device)
    ext.exl3_mgemm(
        x.view(1, m, k),
        trellis_ptrs,
        output,
        suh_ptrs,
        x_had,
        svh_ptrs,
        None,
        None,
        bits,
        -1,
        mcg,
        mul1,
        -1,
        -1,
        0,
        1,
        None,
        None,
    )
    return output

@_exl3_qkv_mgemm.register_fake
def _exl3_qkv_mgemm_fake(
    x: torch.Tensor,
    trellis_ptrs: torch.Tensor,
    suh_ptrs: torch.Tensor,
    svh_ptrs: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
    output_size: int,
) -> torch.Tensor:
    del suh_ptrs, svh_ptrs, bits, mcg, mul1
    return torch.empty(
        (trellis_ptrs.numel(), x.shape[0], output_size),
        dtype=torch.float16,
        device=x.device,
    )

@torch.library.custom_op(
    "vllm::exl3_gemm_bf16_io",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gemm_bf16_io(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
) -> torch.Tensor:
    """Decode-only EXL3 GEMM with a native BF16 serving boundary."""
    ext = _load_exl3_ext()
    m, k = x.shape
    n = trellis.shape[1] * 16
    scratch = torch.empty((m, n), dtype=torch.float16, device=x.device)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    x_had = torch.empty((m, k), dtype=torch.float16, device=x.device)
    ext.exl3_gemm_bf16_io(
        x, trellis, scratch, output, suh, x_had, svh, 2, 160, mcg,
        m > 4,
    )
    return output

@_exl3_gemm_bf16_io.register_fake
def _exl3_gemm_bf16_io_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
) -> torch.Tensor:
    del suh, svh, mcg
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.bfloat16,
        device=x.device,
    )

@torch.library.custom_op(
    "vllm::exl3_mgemm_bf16_io",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_mgemm_bf16_io(
    x: torch.Tensor,
    trellis_ptrs: torch.Tensor,
    suh_ptrs: torch.Tensor,
    svh_ptrs: torch.Tensor,
    unique_suh_ptrs: torch.Tensor,
    had_group_ids: torch.Tensor,
    bits: int,
    output_size: int,
) -> torch.Tensor:
    """Decode-only homogeneous MGEMM writing the final BF16 layout."""
    ext = _load_exl3_ext()
    count = trellis_ptrs.numel()
    m, k = x.shape
    force_num_sms = {2: 85, 8: 20, 14: 12}[count]
    grouped_hadamard = unique_suh_ptrs.numel() < count
    m32_ext = (
        _load_exl3_m32_ext()
        if grouped_hadamard and m == 32 and _BF16_IO_TILE_M32_ENABLED
        else None
    )
    use_tile_m32 = m32_ext is not None
    if grouped_hadamard and m > 16 and not use_tile_m32:
        output = torch.empty(
            (m, count * output_size), dtype=torch.bfloat16, device=x.device
        )
        for start in range(0, m, 16):
            chunk_m = min(16, m - start)
            scratch = torch.empty(
                (count, chunk_m, output_size),
                dtype=torch.float16,
                device=x.device,
            )
            x_had = torch.empty(
                (unique_suh_ptrs.numel(), chunk_m, k),
                dtype=torch.float16,
                device=x.device,
            )
            ext.exl3_mgemm_bf16_io_grouped_had(
                x.narrow(0, start, chunk_m), trellis_ptrs, scratch,
                output.narrow(0, start, chunk_m), unique_suh_ptrs, x_had,
                svh_ptrs, had_group_ids, bits, force_num_sms,
                count * output_size, True,
                True,  # Direct BF16 epilogue is validated for grouped-Hadamard QKV.
                False,  # Every row chunk keeps all matrices resident.
            )
    else:
        scratch = torch.empty(
            (count, m, output_size), dtype=torch.float16, device=x.device
        )
        output = torch.empty(
            (m, count * output_size), dtype=torch.bfloat16, device=x.device
        )
        x_had = torch.empty(
            ((unique_suh_ptrs.numel() if grouped_hadamard else count), m, k),
            dtype=torch.float16,
            device=x.device,
        )
        if grouped_hadamard:
            if use_tile_m32:
                locks = torch.empty(
                    count * (output_size // 128),
                    dtype=torch.int32,
                    device=x.device,
                )
                m32_ext.grouped_had_m32(
                    x, trellis_ptrs, scratch, output, unique_suh_ptrs, x_had,
                    svh_ptrs, had_group_ids, locks, bits, force_num_sms,
                    count * output_size, True,
                )
            else:
                ext.exl3_mgemm_bf16_io_grouped_had(
                    x, trellis_ptrs, scratch, output, unique_suh_ptrs, x_had,
                    svh_ptrs, had_group_ids, bits, force_num_sms,
                    count * output_size, True,
                    True,  # Direct BF16 epilogue is validated for grouped-Hadamard QKV.
                    False,  # All served M<=16 shapes omit the terminal group barrier.
                )
        else:
            ext.exl3_mgemm_bf16_io(
                x, trellis_ptrs, scratch, output, suh_ptrs, x_had, svh_ptrs,
                bits, force_num_sms, count * output_size, True,
                False,  # Wide MLP retains the distributed scratch epilogue.
                False,  # All served M<=16 shapes omit the terminal group barrier.
            )
    return output


@_exl3_mgemm_bf16_io.register_fake
def _exl3_mgemm_bf16_io_fake(
    x: torch.Tensor,
    trellis_ptrs: torch.Tensor,
    suh_ptrs: torch.Tensor,
    svh_ptrs: torch.Tensor,
    unique_suh_ptrs: torch.Tensor,
    had_group_ids: torch.Tensor,
    bits: int,
    output_size: int,
) -> torch.Tensor:
    del suh_ptrs, svh_ptrs, unique_suh_ptrs, had_group_ids, bits
    return torch.empty(
        (x.shape[0], trellis_ptrs.numel() * output_size),
        dtype=torch.bfloat16,
        device=x.device,
    )

def _load_exl3_m32_ext() -> Any | None:
    from .m32 import load_extension
    return load_extension()


def _bf16_bundle_rows_supported(rows: int, bundle: tuple) -> bool:
    if not _BF16_IO_ENABLED:
        return False
    if 1 <= rows <= 16:
        return True
    grouped = bundle[7].numel() < bundle[0].numel()
    return grouped and (
        (rows == 24 and _BF16_IO_M24_ENABLED)
        or (rows == 32 and _BF16_IO_M32_ENABLED)
    )


def _graph_decode_enabled() -> bool:
    """Return whether serialized dense EXL3 may run under CUDA graphs.

    Off by default. ``exl3_gemm`` autotunes with timing launches on the first
    call per shape bucket and those launches fault inside CUDA-graph capture,
    so graphs are only permitted when the operator opts into the pre-capture
    priming pass with ``VLLM_EXL3_GRAPH_DECODE=1``.
    """

    return os.environ.get("VLLM_EXL3_GRAPH_DECODE", "0") == "1"

def _uniform_decode_query_len(vllm_config: Any) -> int:
    """Rows one request contributes to a uniform decode batch."""

    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_spec = getattr(speculative_config, "num_speculative_tokens", None)
    return 1 + int(num_spec) if num_spec else 1

def _graph_decode_capture_rows(vllm_config: Any) -> tuple[int, ...]:
    """Return every row count a decode-only CUDA graph can replay.

    ``cudagraph_capture_sizes`` is already final except for the spec-decode
    alignment that ``CompilationConfig.adjust_cudagraph_sizes_for_spec_decode``
    applies while the KV cache is initialized, i.e. after weights are loaded.
    Priming therefore covers the superset that alignment can produce: every
    configured size, that size rounded up to the uniform decode query length
    (and to the sequence-parallel multiple when that pass binds captured
    sizes), the small interactive request counts alignment always adds, and the
    per-request row counts an MTP/draft layer sees for those sizes.
    """

    compilation_config = getattr(vllm_config, "compilation_config", None)
    sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if not sizes:
        return ()
    configured_max = getattr(compilation_config, "max_cudagraph_capture_size", None)
    max_size = int(configured_max) if configured_max else max(int(s) for s in sizes)
    rows = {int(size) for size in sizes if 0 < int(size) <= max_size}
    query_len = _uniform_decode_query_len(vllm_config)
    if query_len > 1:
        multiples = [query_len]
        pass_config = getattr(compilation_config, "pass_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        if tp_size > 1 and getattr(pass_config, "enable_sp", False):
            multiples.append(max(query_len, tp_size))
        for multiple in multiples:
            for size in tuple(rows):
                rounded = -(-size // multiple) * multiple
                if rounded <= max_size:
                    rows.add(rounded)
        rows.update(
            query_len * requests
            for requests in range(1, 33)
            if query_len * requests <= max_size
        )
        # A draft/MTP layer consumes one row per request, not per drafted token.
        rows.update(size // query_len for size in tuple(rows) if size >= query_len)
    return tuple(sorted(rows))

def _prime_exl3_gemm_rows(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    has_mcg: bool,
    has_mul1: bool,
    rows: tuple[int, ...],
    owner: str,
) -> None:
    """Autotune one serialized shard for every capturable row count.

    exllamav3_ext hashes its autotune cache over the m bucket, k, n, K and the
    codebook selector, so one zero-filled eager launch per row count here
    removes every timing launch the same shape would otherwise attempt inside
    CUDA-graph capture, and materializes the extension's per-device lock arena
    outside capture. This is the serialized counterpart of
    ``Exl3OnlineLinearMethod._warm_decode_shapes``.
    """

    device = trellis.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    k = int(trellis.shape[0]) * 16
    n = int(trellis.shape[1]) * 16
    bits = int(trellis.shape[2]) // 16
    # Mirrors the extension's codebook selector: mcg=1, mul1=2, otherwise 0.
    codebook = 1 if has_mcg else 2 if has_mul1 else 0
    pending = [
        int(m)
        for m in rows
        if (device_index, int(m), k, n, bits, codebook)
        not in _EXL3_GEMM_PRIMED_SIGNATURES
    ]
    if not pending:
        return
    # One arena for the whole shape: a leading-row view of a contiguous buffer
    # is itself contiguous, so the kernel contract holds without reallocating
    # per row count.
    source = torch.zeros((max(pending), k), dtype=torch.float16, device=device)
    for m in pending:
        try:
            _exl3_gemm(
                source.narrow(0, 0, m),
                trellis,
                suh,
                svh,
                has_mcg,
                has_mul1,
            )
        except Exception as exc:
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True) or unset "
                "VLLM_EXL3_GRAPH_DECODE. exl3_gemm autotuning could not be "
                f"primed for {owner} at m={m}, K={k}, N={n}, bits={bits}, so "
                "CUDA-graph capture of that shape would fault."
            ) from exc
        _EXL3_GEMM_PRIMED_SIGNATURES.add((device_index, m, k, n, bits, codebook))
    torch.cuda.synchronize(device)
    logger.info_once(
        "EXL3 graph-decode priming: autotuned exl3_gemm for %d capture row "
        "counts (m=%d..%d) at K=%d, N=%d, bits=%d, codebook=%d.",
        len(pending),
        pending[0],
        pending[-1],
        k,
        n,
        bits,
        codebook,
    )

def _b12x_trellis_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    dtype: torch.dtype,
) -> Any:
    """Prepare each online weight before capture and retain its native views."""

    api = _load_b12x_trellis_linear()
    # Keep prepared views on the owning tensor. A process-global pointer-keyed
    # cache can alias after allocator address reuse and retains released models
    # forever. This small reference cycle is collected with the tensor.
    cache = getattr(trellis, "_vllm_b12x_prepared_weights", None)
    if cache is None:
        cache = {}
        trellis._vllm_b12x_prepared_weights = cache
    key = (id(suh), id(svh), dtype)
    weight = cache.get(key)
    if weight is None:
        weight = api.prepare_weight(
            trellis,
            suh,
            svh,
            codebook="mcg",
            params_dtype=dtype,
        )
        cache[key] = weight
    return weight

def _b12x_trellis_n_bounds() -> tuple[int, int] | None:
    """Packed-N window the B12X K6 route should serve; outside it use exl3_gemm.

    The bounded default avoids known small-M regressions for very wide output
    heads and tiny-N projections. Eager call sites also pay B12X's Python
    dispatch overhead, which the thin exl3_gemm binding avoids.

    VLLM_EXL3_B12X_N_RANGE is "<lo>-<hi>" in output features; "0" or empty
    restores the unbounded pre-window behaviour.
    """

    raw = os.environ.get("VLLM_EXL3_B12X_N_RANGE", "5120-32768").strip()
    if raw in ("", "0"):
        return None
    lo, _, hi = raw.partition("-")
    try:
        return int(lo), int(hi)
    except ValueError as exc:
        raise ValueError(
            f"VLLM_EXL3_B12X_N_RANGE must be '<lo>-<hi>' or '0', got {raw!r}"
        ) from exc

def _b12x_trellis_k6_supported(
    trellis: torch.Tensor,
    *,
    has_mcg: bool,
    has_mul1: bool,
) -> bool:
    """Gate the native path to the contract b12x's trellis256 kernel implements.

    B12X accepts 3/4/5/6-bit payloads and `api.prepare_weight` infers the width
    from the tensor. The default remains K6-only; the explicit opt-in admits
    the additional encoded widths after their plans have been warmed.

    VLLM_EXL3_B12X_ANY_BITS=1 admits 3/4/5/6-bit; default stays K6-only.
    """
    # When VLLM_EXL3_SKIP_TRELLIS_PREP=1, skip b12x prepared weights (saves 16GB).
    # Attention layers fall back to ext.exl3_gemm (raw trellis codes, no prep needed).
    if os.environ.get("VLLM_EXL3_SKIP_TRELLIS_PREP", "0") == "1":
        return False
    bounds = _b12x_trellis_n_bounds()
    if bounds is not None:
        n_packed = int(trellis.shape[1]) * 16
        if not bounds[0] <= n_packed <= bounds[1]:
            return False
    n_words = int(trellis.shape[2])
    if os.environ.get("VLLM_EXL3_B12X_ANY_BITS", "0") == "1":
        # Each encoded width owns a separate B12X plan. Warm every (device,
        # bits) pair before graph capture so no plan allocates graph-private
        # buffers on its first real invocation.
        bits_ok = n_words in (48, 64, 80, 96)
    else:
        bits_ok = n_words == 96
    return bool(
        has_mcg
        and not has_mul1
        and trellis.dtype == torch.int16
        and trellis.ndim == 3
        and bits_ok
        and int(trellis.shape[0]) % 8 == 0
        and int(trellis.shape[1]) % 8 == 0
    )

def _warm_b12x_trellis_device(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> None:
    """Build the runtime extension before any CUDA graph starts capturing.

    Keyed on (device, bits), not device alone. B12X compiles and initializes a
    separate mixed-Trellis plan per encoded width. Warming every plan before
    capture prevents first-use allocations from entering a graph-private pool.
    """

    device_index = trellis.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    bits = int(trellis.shape[2]) // 16
    key = (device_index, bits)
    if key in _B12X_TRELLIS_WARMED_DEVICES:
        return
    source = torch.zeros(
        (1, int(trellis.shape[0]) * 16),
        dtype=torch.float16,
        device=trellis.device,
    )
    _b12x_trellis_linear(source, trellis, suh, svh)
    torch.cuda.synchronize(trellis.device)
    _B12X_TRELLIS_WARMED_DEVICES.add(key)

def _b12x_trellis_c_tmp_elements(
    rows: int, columns: int, *, bits: int = 6
) -> int:
    """Return graph-safe dense-W4A16 scratch capacity for one static shape."""

    rows = int(rows)
    columns = int(columns)
    bits = int(bits)
    if rows <= 128 and bits == 6:
        # The cooperative K6 small-M kernel does not consume W4A16 scratch.
        return 1
    # Non-K6 payloads (this checkpoint's K5 mlp.gate_proj/up_proj) take the
    # scratch-consuming path even at decode row counts, so the K6 shortcut
    # above would hand b12x a 1-element buffer and it raises
    # "W4A16 GEMM scratch is not initialized for CUDA graph capture" during
    # capture. Mirror b12x's own sizing rather than guessing: the dense runner
    # rounds m UP to a multiple of the routed block size
    # (route_slots = ceil(m/block)*block, kernel.py _run_trellis256_dense_*)
    # before calling packed_gemm_scratch_elements, so sizing on raw `rows`
    # under-allocates by up to 64x. Cover every allowed block size.
    try:
        from b12x.moe._shared.kernels.w4a16.host import (
            _W4A16_ALLOWED_ROUTED_SIZES,
            packed_gemm_scratch_elements,
        )

        sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        need = max(
            packed_gemm_scratch_elements(
                size_n=columns,
                route_slots=((rows + block - 1) // block) * block,
                moe_block_size=block,
                sms=int(sms),
            )
            for block in _W4A16_ALLOWED_ROUTED_SIZES
        )
        return min(int(need), _B12X_TRELLIS_C_TMP_CAP)
    except Exception:
        pass
    padded_rows = max(
        ((rows + 47) // 48) * 48,
        ((rows + 63) // 64) * 64,
    )
    return min(columns * padded_rows, _B12X_TRELLIS_C_TMP_CAP)

def _b12x_trellis_c_tmp_shared(device: torch.device) -> torch.Tensor:
    """Return the one scratch accumulator shared by every B12X matrix.

    ``packed_gemm_scratch_elements`` returns
    ``min(size_n * route_slots, sms * 4 * moe_block_size * 256)``, and on this
    card the right-hand cap binds for every matrix in the checkpoint, so the
    requirement is *shape independent*: one buffer at the cap (~11.1M fp32
    elements, 42.5 MiB on 170 SMs) satisfies the largest legal request.  c_tmp
    is a transient GEMM accumulator, so sharing it across matrices is safe -
    calls inside a forward pass are serialised on one stream.

    Sizing it per (m, k, n, bits) shape - which is what the buffer cache below
    used to do - pays that 42.5 MiB once per *distinct shape*.  Across 409
    matrices with a 3072-row prefill chunk that reached tens of GiB and OOM'd
    the engine on the first real prefill, which is why B12X prep looked
    unusable for prefill.

    Allocated once and never grown, so a pointer captured into a CUDA graph
    stays valid for the process lifetime.
    """

    idx = -1 if device.index is None else int(device.index)
    buf = _B12X_C_TMP_SHARED.get(idx)
    if buf is not None:
        return buf
    need = _B12X_TRELLIS_C_TMP_CAP
    try:
        from b12x.moe._shared.kernels.w4a16.host import (
            _W4A16_ALLOWED_ROUTED_SIZES,
        )

        sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        need = min(
            need,
            max(
                sms * 4 * int(block) * 256 * (2 if int(block) == 8 else 1)
                for block in _W4A16_ALLOWED_ROUTED_SIZES
            ),
        )
    except Exception:
        pass
    buf = torch.empty((int(need),), dtype=torch.float32, device=device)
    _B12X_C_TMP_SHARED[idx] = buf
    return buf

@torch.library.custom_op(
    "vllm::b12x_trellis_linear_out",
    mutates_args=("output", "gemm_output", "c_tmp", "rotated_f16"),
    device_types="cuda",
)
def _b12x_trellis_linear_out(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    output: torch.Tensor,
    gemm_output: torch.Tensor,
    c_tmp: torch.Tensor,
    rotated_f16: torch.Tensor,
) -> None:
    """Execute dense Trellis into graph-owned output and scratch tensors."""

    api = _load_b12x_trellis_linear()
    weight = _b12x_trellis_weight(trellis, suh, svh, x.dtype)
    api.run(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
        rotated_f16=rotated_f16,
    )

@_b12x_trellis_linear_out.register_fake
def _b12x_trellis_linear_out_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    output: torch.Tensor,
    gemm_output: torch.Tensor,
    c_tmp: torch.Tensor,
    rotated_f16: torch.Tensor,
) -> None:
    del x, trellis, suh, svh, output, gemm_output, c_tmp, rotated_f16

def _b12x_trellis_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Run every online-K6 batch through B12X with cached storage."""
    m = x.shape[0]
    n = trellis.shape[1] * 16
    bits = int(trellis.shape[2]) // 16
    c_tmp = _b12x_trellis_c_tmp_shared(x.device)
    if m <= _B12X_TRELLIS_BUF_CACHE_MAX_ROWS:
        key = (x.shape[0], x.shape[1], n, bits, x.dtype, x.device.index)
        if not hasattr(_b12x_trellis_linear, '_buf_cache'):
            _b12x_trellis_linear._buf_cache = {}  # type: ignore[attr-defined]
        buf = _b12x_trellis_linear._buf_cache.get(key)  # type: ignore[attr-defined]
        if buf is None:
            output = torch.empty(m, n, dtype=x.dtype, device=x.device)
            buf = (output, torch.empty_like(output), torch.empty_like(x))
            _b12x_trellis_linear._buf_cache[key] = buf  # type: ignore[attr-defined]
        output, gemm_output, rotated_f16 = buf
    else:
        # Prefill: let the caching allocator recycle these.  Retaining one
        # m x n pair per distinct chunk shape is what exhausted the GPU.
        output = torch.empty(m, n, dtype=x.dtype, device=x.device)
        gemm_output = torch.empty_like(output)
        rotated_f16 = torch.empty_like(x)
    _b12x_trellis_linear_out(
        x, trellis, suh, svh,
        output, gemm_output, c_tmp, rotated_f16,
    )
    return output

class Exl3Config(QuantizationConfig):
    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        codebook: str | None = None,
        version: str | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.version = version
        self.tensor_storage = tensor_storage or {}
        self._eager_checked = False
        self.graph_decode_rows: tuple[int, ...] | None = None

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The kernel boundary is always fp16.  BF16 model activations are cast
        # in apply() and converted back after the fp16 bias addition.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Runtime coverage for this release stops at SM120. The underlying
        # extension may compile for older architectures, but build success is
        # not a compatibility claim.
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        instance = cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            tensor_storage=config.get("tensor_storage"),
        )
        return instance

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> str | None:
        del cls, hf_config
        if user_quant is not None and user_quant != "exl3":
            return None
        return "exl3" if hf_quant_cfg.get("tensor_storage") else None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        if not self.tensor_storage:
            resolved_revision = revision
            if resolved_revision is None and hf_config is not None:
                resolved_revision = getattr(hf_config, "_commit_hash", None)
            config = get_hf_file_to_dict(
                "quantization_config.json",
                model_name,
                revision=resolved_revision,
            )
            if not config or not config.get("tensor_storage"):
                raise ValueError(
                    "EXL3 requires quantization_config.json with a non-empty "
                    "tensor_storage map. For branch-indexed Hugging Face repos, "
                    "download/serve an actual bpw revision rather than main."
                )
            self.bits = config.get("bits", self.bits)
            self.head_bits = config.get("head_bits", self.head_bits)
            self.codebook = config.get("codebook", self.codebook)
            self.version = config.get("version", self.version)
            self.tensor_storage = config["tensor_storage"]
        self._validate_storage_metadata()
        self._validate_model_config(hf_config)
        mxfp6_hybrid.validate_profile_model(hf_config)
        self._force_independent_lm_head(hf_config)

    @staticmethod
    def _validate_model_config(hf_config: PretrainedConfig | None) -> None:
        if hf_config is None:
            return
        try:
            config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            config = hf_config
        model_type = getattr(config, "model_type", None)
        architectures = set(getattr(config, "architectures", None) or ())
        hidden_size = getattr(config, "hidden_size", None)
        problems: list[str] = []
        if model_type not in _SUPPORTED_MODEL_TYPES:
            problems.append(f"model_type={model_type!r}")
        if architectures and _SUPPORTED_ARCHITECTURE not in architectures:
            problems.append(f"architectures={sorted(architectures)!r}")
        if hidden_size != _SUPPORTED_HIDDEN_SIZE:
            problems.append(f"hidden_size={hidden_size!r}")
        if problems:
            raise ValueError(
                "vLLM Mach 0.1.0 has only validated the Qwen3.8-27B Dense "
                "layout (Qwen3_5ForConditionalGeneration, hidden_size=5120); "
                "refusing unverified configuration: " + ", ".join(problems)
            )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        # Keep both spellings: loader prefixes use vLLM names, while packed
        # source-matrix discovery intentionally refers to the unstacked HF name.
        mapped = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
        self.tensor_storage = {**self.tensor_storage, **mapped}

    def _validate_storage_metadata(self) -> None:
        bad: list[str] = []
        exl3_count = 0
        for prefix, entry in self.tensor_storage.items():
            if entry.get("quant_format") != "exl3":
                continue
            exl3_count += 1
            stored = entry.get("stored_tensors", {})
            suffixes = {name.rsplit(".", 1)[-1] for name in stored}
            required = {"trellis"}
            if not ({"suh", "su"} & suffixes):
                required.add("suh|su")
            if not ({"svh", "sv"} & suffixes):
                required.add("svh|sv")
            missing = [name for name in required if name not in suffixes]
            if missing:
                bad.append(f"{prefix}: missing {','.join(sorted(missing))}")
            if {"mcg", "mul1"} <= suffixes:
                bad.append(f"{prefix}: both mcg and mul1 are present")
        if not exl3_count:
            raise ValueError("quantization_config.json has no EXL3 tensor records")
        if bad:
            raise ValueError("Invalid EXL3 tensor metadata: " + "; ".join(bad[:16]))

    def _force_independent_lm_head(self, hf_config: PretrainedConfig | None) -> None:
        if hf_config is None or not self.has_quantized_lm_head():
            return
        configs: list[Any] = [hf_config]
        try:
            text_config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None and text_config is not hf_config:
            configs.append(text_config)
        changed = False
        for config in configs:
            if getattr(config, "tie_word_embeddings", False):
                config.tie_word_embeddings = False
                changed = True
        if changed:
            logger.warning_once(
                "EXL3 metadata contains an independently quantized lm_head; "
                "overriding tie_word_embeddings so vLLM instantiates it."
            )

    def _graph_decode_refusal(self, vllm_config: Any) -> str | None:
        """Return why graph decode is refused for this run, or None to allow.

        Only decode-only capture is admissible. The capture-size list bounds
        every row count a decode graph replays, so those shapes can be primed
        before capture, while a mode that also captures mixed prefill batches
        would autotune inside capture at token counts the scheduler picks at
        runtime.
        """

        if not _graph_decode_enabled():
            return (
                "VLLM_EXL3_GRAPH_DECODE is not 1, so the pre-capture exl3_gemm "
                "priming pass is disabled"
            )
        compilation_config = getattr(vllm_config, "compilation_config", None)
        mode = getattr(compilation_config, "cudagraph_mode", None)
        if mode is None:
            return "compilation_config.cudagraph_mode is unset"
        if not bool(mode):
            # CUDAGraphMode.NONE never captures, so nothing needs priming.
            return None
        if mode.mixed_mode() != CUDAGraphMode.NONE:
            return (
                f"cudagraph_mode={mode} also captures mixed prefill batches, "
                "whose token counts are not enumerable before capture; select "
                "decode-only capture with "
                "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"
            )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if getattr(parallel_config, "use_ubatching", False):
            return (
                "microbatched execution (DBO/ubatching) splits every captured "
                "size across ubatches, so the row counts reaching a shard are "
                "not the capture sizes this priming pass covers"
            )
        rows = _graph_decode_capture_rows(vllm_config)
        if not rows:
            return (
                f"cudagraph_mode={mode} is decode-only but "
                "compilation_config.cudagraph_capture_sizes is empty, so no "
                "row count can be primed"
            )
        self.graph_decode_rows = rows
        return None

    def _require_enforce_eager(self) -> None:
        # MULTIPRECISION normally routes every layer through b12x dense_gemm
        # or _b12x_trellis_linear, so exl3_gemm priming is unnecessary.  The
        # optional QKV bundle deliberately reintroduces exl3_mgemm for decode
        # and must continue through the graph-safety check below.
        if self._eager_checked:
            return
        self._eager_checked = True
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
        if (tp_size, pp_size) != (2, 1):
            raise ValueError(
                "vLLM Mach 0.1.0 has only validated TP2 with PP1; "
                f"found TP{tp_size} with PP{pp_size}."
            )
        if vllm_config.model_config.enforce_eager:
            return
        refusal = self._graph_decode_refusal(vllm_config)
        if refusal is not None:
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True). exl3_gemm "
                "autotunes with timing launches on first use per shape "
                "bucket, which is incompatible with CUDA-graph capture. "
                f"Graph decode was not permitted because {refusal}."
            )
        if self.graph_decode_rows:
            logger.info_once(
                "EXL3 graph decode enabled by VLLM_EXL3_GRAPH_DECODE: "
                "cudagraph_mode=%s captures decode only; priming exl3_gemm for "
                "%d row counts (m=%d..%d) during weight loading.",
                vllm_config.compilation_config.cudagraph_mode,
                len(self.graph_decode_rows),
                self.graph_decode_rows[0],
                self.graph_decode_rows[-1],
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        self._require_enforce_eager()
        if (
            type(layer).__name__ == "VocabParallelEmbedding"
            and (embed_bits := _embed_online_bits()) is not None
        ):
            return Exl3OnlineEmbeddingMethod(embed_bits)
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if self._linear_prefix_is_exl3(prefix):
                return Exl3LinearMethod(self)
            return UnquantizedLinearMethod()
        return None

    def _storage_entry(self, prefix: str) -> dict[str, Any] | None:
        candidates = [prefix]
        if prefix.startswith("model."):
            candidates.append(prefix.removeprefix("model."))
        else:
            candidates.append(f"model.{prefix}")

        # Multimodal wrappers often add an extra `model` or `language_model`
        # segment relative to vLLM's text-only module — interior
        # (`model.language_model.layers...`) or leading
        # (`language_model.lm_head`), so leading segments collapse too.
        parts = prefix.split(".")
        for removable in ("model", "language_model"):
            for idx in range(0, len(parts) - 1):
                if parts[idx] != removable:
                    continue
                collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
                candidates.extend((collapsed, f"model.{collapsed}"))
                if collapsed.startswith("model."):
                    candidates.append(collapsed.removeprefix("model."))

        for candidate in dict.fromkeys(candidates):
            entry = self.tensor_storage.get(candidate)
            if entry is not None:
                return entry
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = self._storage_entry(prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if self._is_exl3_prefix(prefix):
            return True
        leaf = prefix.rsplit(".", 1)[-1]
        source_leaves = self.packed_modules_mapping.get(leaf)
        if not source_leaves:
            return False
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        source_is_exl3 = [
            self._is_exl3_prefix(f"{base}.{source}" if base else source)
            for source in source_leaves
        ]
        if any(source_is_exl3) and not all(source_is_exl3):
            raise ValueError(
                f"Packed EXL3 projection {prefix!r} mixes EXL3 and BF16 "
                "source shards; a fused module must use one quantization "
                "scheme."
            )
        return all(source_is_exl3)

    def codebook_for_prefix(self, prefix: str) -> str | None:
        entry = self._storage_entry(prefix)
        if entry is None:
            return None
        suffixes = {name.rsplit(".", 1)[-1] for name in entry.get("stored_tensors", {})}
        if "mcg" in suffixes:
            return "mcg"
        if "mul1" in suffixes:
            return "mul1"
        return None

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")


class Exl3Parameter(BasevLLMParameter):
    """Zero-sized parameter holding independently shaped EXL3 components."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[ShardId, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        if getattr(loaded_weight, "_vllm_instanttensor_borrowed", False):
            loaded_weight = loaded_weight.clone()
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()

def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: ShardId = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)

class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if layer.__class__.__name__ == "ParallelLMHead":
            org = getattr(layer, "org_vocab_size", None)
            total = getattr(layer, "num_embeddings", None)
            if org is not None and total is not None and org != total:
                raise NotImplementedError(
                    "EXL3 lm_head with added vocabulary is unsupported: the "
                    f"trellis tensor covers the original {org} rows but the "
                    f"layer allocates {total}; TP slicing would silently "
                    "misalign. Strip --lora-extra-vocab-size / added tokens "
                    "or leave lm_head unquantized."
                )
        # Respect the layer's effective topology. disable_tp linears set their
        # own tp_size=1, while ReplicatedLinear carries full weights even when
        # the process-wide tensor group is larger than one.
        if isinstance(layer, ReplicatedLinear):
            layer.exl3_tp_rank = 0
            layer.exl3_tp_size = 1
        else:
            layer.exl3_tp_rank = getattr(
                layer, "tp_rank", get_tensor_model_parallel_rank()
            )
            layer.exl3_tp_size = getattr(
                layer, "tp_size", get_tensor_model_parallel_world_size()
            )
        layer.exl3_input_size = input_size
        layer.exl3_input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = (
            "row" if input_size_per_partition != input_size else "column"
        )
        source_prefixes = self._source_prefixes_for_layer(layer, layer.exl3_shard_ids)
        layer.exl3_expected_codebooks = {
            shard_id: self.quant_config.codebook_for_prefix(source_prefix)
            for shard_id, source_prefix in zip(
                layer.exl3_shard_ids, source_prefixes, strict=True
            )
        }

        # su/sv are legacy packed sign bitfields.  Modern checkpoints load
        # suh/svh directly.
        for name in ("suh", "svh", "su", "sv", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._materialize_legacy_hadamard(layer)
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        for shard_id in layer.exl3_shard_ids:
            expected = layer.exl3_expected_codebooks[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if has_mcg and has_mul1:
                missing.append(f"codebook[{shard_id!r}]=both mcg and mul1")
            elif expected == "mcg" and not has_mcg:
                missing.append(f"mcg[{shard_id!r}]")
            elif expected == "mul1" and not has_mul1:
                missing.append(f"mul1[{shard_id!r}]")
            elif expected is None and (has_mcg or has_mul1):
                missing.append(f"unexpected codebook[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(
                f"Missing or inconsistent EXL3 tensors for {prefix}: "
                + ", ".join(missing)
            )
        self._validate_loaded_tensors(layer)
        self._shard_tensors_for_tensor_parallel(layer)
        self._validate_loaded_tensors(layer)
        device = layer.trellis.device
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(
                    device=device, non_blocking=True
                ).contiguous()
        hybrid = None
        prefix = str(getattr(layer, "prefix", ""))
        if mxfp6_hybrid.route_for_prefix(prefix) is not None:
            hybrid = mxfp6_hybrid.prepare_layer(layer, _load_exl3_ext())
        if hybrid is not None and hybrid.route is mxfp6_hybrid.HybridRoute.ALL_ROWS:
            return
        self._prepare_qkv_mgemm(layer)
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            if not _b12x_trellis_k6_supported(
                trellis,
                has_mcg=shard_id in layer.mcg.exl3_tensors,
                has_mul1=shard_id in layer.mul1.exl3_tensors,
            ):
                continue
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            _b12x_trellis_weight(
                trellis,
                suh,
                svh,
                torch.float16,
            )
            _warm_b12x_trellis_device(trellis, suh, svh)
        self._prime_graph_decode_shapes(layer)

    def _prime_graph_decode_shapes(self, layer: torch.nn.Module) -> None:
        """Autotune every capturable decode shape while still executing eagerly.

        This is the serialized counterpart of
        ``Exl3OnlineLinearMethod._warm_decode_shapes``: the online path warms
        rows 1..6 because that is its entire decode window, whereas a captured
        decode graph replays exactly the configured capture sizes. No-op unless
        ``_require_enforce_eager`` granted graph decode.
        """
        rows = self.quant_config.graph_decode_rows
        if not rows:
            return
        owner = getattr(layer, "prefix", layer.__class__.__name__)
        bundle = getattr(layer, "exl3_qkv_mgemm", None)
        if bundle is not None:
            device = layer.trellis.device
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            k = layer.exl3_input_size_per_partition
            output_size = bundle[6]
            codebook = 1 if bundle[4] else 2 if bundle[5] else 0
            count = bundle[0].numel()
            pending = tuple(
                int(m)
                for m in rows
                if (
                    device_index,
                    int(m),
                    k,
                    output_size,
                    bundle[3],
                    codebook,
                    count,
                )
                not in _EXL3_QKV_MGEMM_PRIMED_SIGNATURES
            )
            if not pending:
                return
            source = torch.zeros(
                (max(pending), k),
                dtype=torch.float16,
                device=device,
            )
            for m in pending:
                if _bf16_bundle_rows_supported(m, bundle):
                    _exl3_mgemm_bf16_io(
                        source.narrow(0, 0, m).to(torch.bfloat16),
                        bundle[0],
                        bundle[1],
                        bundle[2],
                        bundle[7],
                        bundle[8],
                        bundle[3],
                        bundle[6],
                    )
                else:
                    _exl3_qkv_mgemm(
                        source.narrow(0, 0, m),
                        bundle[0],
                        bundle[1],
                        bundle[2],
                        bundle[3],
                        bundle[4],
                        bundle[5],
                        bundle[6],
                    )
                _EXL3_QKV_MGEMM_PRIMED_SIGNATURES.add(
                    (
                        device_index,
                        m,
                        k,
                        output_size,
                        bundle[3],
                        codebook,
                        count,
                    )
                )
            torch.cuda.synchronize(device)
            logger.info(
                "EXL3 QKV MGEMM graph-decode priming: warmed %d capture row "
                "counts (m=%d..%d), K=%d, N=%d, matrices=%d.",
                len(pending),
                pending[0],
                pending[-1],
                k,
                output_size,
                count,
            )
            return
        prefix = getattr(layer, "prefix", "")
        if _BF16_IO_ENABLED and prefix.endswith(
            ("mlp.down_proj", "linear_attn.out_proj", "self_attn.o_proj")
        ):
            if len(layer.exl3_shard_ids) != 1:
                raise ValueError(
                    f"EXL3 BF16-I/O priming expected one shard for {prefix}"
                )
            shard_id = layer.exl3_shard_ids[0]
            trellis = layer.trellis.exl3_tensors[shard_id]
            source = torch.zeros(
                (max(m for m in rows if m <= 16), trellis.shape[0] * 16),
                dtype=torch.bfloat16,
                device=trellis.device,
            )
            for m in rows:
                if m <= 16:
                    _exl3_gemm_bf16_io(
                        source.narrow(0, 0, m),
                        trellis,
                        layer.suh.exl3_tensors[shard_id],
                        layer.svh.exl3_tensors[shard_id],
                        shard_id in layer.mcg.exl3_tensors,
                    )
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if _b12x_trellis_k6_supported(
                trellis,
                has_mcg=has_mcg,
                has_mul1=has_mul1,
            ):
                # The native K6 path picks its kernel from the shape alone and
                # is already prepared and warmed above.
                continue
            _prime_exl3_gemm_rows(
                trellis,
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                has_mcg=has_mcg,
                has_mul1=has_mul1,
                rows=rows,
                owner=f"{owner}[{shard_id!r}]",
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        hybrid = mxfp6_hybrid.state_for_rows(layer, rows)
        bundle = getattr(layer, "exl3_qkv_mgemm", None)
        prefix = getattr(layer, "prefix", "")
        small_bf16 = (
            _BF16_IO_ENABLED
            and bias is None
            and original_dtype == torch.bfloat16
            and x.reshape(-1, x.shape[-1]).shape[0] <= 16
        )
        bf16_regular = small_bf16 and prefix.endswith(
            ("mlp.down_proj", "linear_attn.out_proj", "self_attn.o_proj")
        )
        bf16_bundle = (
            bundle is not None
            and bias is None
            and original_dtype == torch.bfloat16
            and _bf16_bundle_rows_supported(rows, bundle)
        )
        target_dtype = (
            torch.bfloat16
            if hybrid
            else (torch.bfloat16 if bf16_regular or bf16_bundle else torch.float16)
        )
        x_2d = x.reshape(-1, x.shape[-1]).to(target_dtype).contiguous()
        if hybrid is not None and hybrid.merged_weight is not None:
            output = mxfp6_hybrid.apply_weight(x_2d, hybrid.merged_weight)
        elif hybrid is not None:
            outputs = [
                mxfp6_hybrid.apply_weight(x_2d, hybrid.weights[shard_id])
                for shard_id in layer.exl3_shard_ids
            ]
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        elif bf16_bundle:
            output = _exl3_mgemm_bf16_io(
                x_2d,
                bundle[0],
                bundle[1],
                bundle[2],
                bundle[7],
                bundle[8],
                bundle[3],
                bundle[6],
            )
        elif bundle is not None and x_2d.shape[0] < 128:
            stacked = _exl3_qkv_mgemm(
                x_2d,
                bundle[0],
                bundle[1],
                bundle[2],
                bundle[3],
                bundle[4],
                bundle[5],
                bundle[6],
            )
            output = torch.cat(stacked.unbind(0), dim=-1)
        elif bf16_regular:
            if len(layer.exl3_shard_ids) != 1:
                raise ValueError(
                    f"EXL3 BF16-I/O regular path expected one shard for {prefix}"
                )
            shard_id = layer.exl3_shard_ids[0]
            output = _exl3_gemm_bf16_io(
                x_2d,
                layer.trellis.exl3_tensors[shard_id],
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                shard_id in layer.mcg.exl3_tensors,
            )
        else:
            outputs = [
                self._apply_one(layer, x_2d, shard_id)
                for shard_id in layer.exl3_shard_ids
            ]
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    @staticmethod
    def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
        words = bitfield.contiguous().view(torch.uint16).to(torch.int32)
        masks = 1 << torch.arange(16, device=words.device, dtype=torch.int32)
        negative = (words.unsqueeze(-1) & masks) != 0
        return (
            torch.where(
                negative,
                torch.tensor(-1.0, device=words.device, dtype=torch.float16),
                torch.tensor(1.0, device=words.device, dtype=torch.float16),
            )
            .flatten()
            .contiguous()
        )

    @classmethod
    def _materialize_legacy_hadamard(cls, layer: torch.nn.Module) -> None:
        for packed_name, half_name in (("su", "suh"), ("sv", "svh")):
            packed = getattr(layer, packed_name).exl3_tensors
            half = getattr(layer, half_name).exl3_tensors
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in half and shard_id in packed:
                    half[shard_id] = cls._unpack_signs(packed[shard_id])

    @classmethod
    def _prepare_qkv_mgemm(cls, layer: torch.nn.Module) -> None:
        """Build the two Qwen3.8-27B TP2 decode projection bundles.

        The extension requires a common K, N, bit width, and codebook.  Wide
        projections are therefore copied into fixed-width, output-contiguous
        trellis slices once at model load.  Original tensors remain authoritative
        for prefill and for the disabled-control path.
        """
        if not _QKV_MGEMM_ENABLED:
            return
        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("in_proj_qkvz"):
            shard_ids: tuple[ShardId, ...] = (0, 1, 2, 3)
            expected_sizes = (1024, 1024, 3072, 3072)
            output_size = 1024
        elif isinstance(layer, QKVParallelLinear):
            shard_ids = ("q", "k", "v")
            expected_sizes = (6144, 512, 512)
            output_size = 512
        elif _BF16_IO_ENABLED and prefix.endswith("mlp.gate_up_proj"):
            shard_ids = (0, 1)
            expected_sizes = (8704, 8704)
            output_size = 8704
        else:
            return

        actual_sizes = tuple(
            int(layer.trellis.exl3_tensors[sid].shape[1]) * 16
            for sid in shard_ids
        )
        if (
            layer.exl3_tp_size != 2
            or layer.exl3_input_size_per_partition != 5120
            or tuple(layer.exl3_shard_ids) != shard_ids
            or actual_sizes != expected_sizes
        ):
            raise ValueError(
                "EXL3_QKV_MGEMM only supports the validated Qwen3.8-27B TP2 "
                f"geometry; {prefix} has tp={layer.exl3_tp_size}, "
                f"K={layer.exl3_input_size_per_partition}, "
                f"shards={tuple(layer.exl3_shard_ids)!r}, N={actual_sizes}."
            )

        trellis_chunks: list[torch.Tensor] = []
        suh_chunks: list[torch.Tensor] = []
        svh_chunks: list[torch.Tensor] = []
        copied_bytes = 0
        signatures: set[tuple[int, bool, bool]] = set()
        for shard_id, shard_size in zip(shard_ids, expected_sizes, strict=True):
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            signatures.add((int(trellis.shape[2]) // 16, has_mcg, has_mul1))
            for start in range(0, shard_size, output_size):
                if shard_size == output_size:
                    trellis_chunk = trellis
                    svh_chunk = svh
                else:
                    trellis_chunk = trellis.narrow(
                        1, start // 16, output_size // 16
                    ).contiguous()
                    svh_chunk = svh.narrow(0, start, output_size).contiguous()
                    copied_bytes += (
                        trellis_chunk.numel() * trellis_chunk.element_size()
                        + svh_chunk.numel() * svh_chunk.element_size()
                    )
                trellis_chunks.append(trellis_chunk)
                suh_chunks.append(suh)
                svh_chunks.append(svh_chunk)
        expected_count = sum(size // output_size for size in expected_sizes)
        if len(signatures) != 1 or len(trellis_chunks) != expected_count:
            raise ValueError(
                f"EXL3_QKV_MGEMM requires {expected_count} homogeneous matrices; "
                f"{prefix} produced {len(trellis_chunks)} with {signatures}."
            )
        bits, has_mcg, has_mul1 = signatures.pop()
        device = layer.trellis.device
        trellis_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in trellis_chunks],
            dtype=torch.long,
            device=device,
        )
        suh_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in suh_chunks],
            dtype=torch.long,
            device=device,
        )
        svh_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in svh_chunks],
            dtype=torch.long,
            device=device,
        )
        unique_suh_chunks: list[torch.Tensor] = []
        had_group_ids: list[int] = []
        group_by_ptr: dict[int, int] = {}
        for tensor in suh_chunks:
            ptr = tensor.data_ptr()
            group = group_by_ptr.get(ptr)
            if group is None:
                group = len(unique_suh_chunks)
                group_by_ptr[ptr] = group
                unique_suh_chunks.append(tensor)
            had_group_ids.append(group)
        unique_suh_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in unique_suh_chunks],
            dtype=torch.long,
            device=device,
        )
        had_group_ids_tensor = torch.tensor(
            had_group_ids, dtype=torch.int32, device=device
        )
        # Keep every pointed-to tensor alive for the lifetime of the layer.
        layer.exl3_qkv_mgemm_tensors = (
            tuple(trellis_chunks),
            tuple(suh_chunks),
            tuple(svh_chunks),
            tuple(unique_suh_chunks),
        )
        layer.exl3_qkv_mgemm = (
            trellis_ptrs,
            suh_ptrs,
            svh_ptrs,
            bits,
            has_mcg,
            has_mul1,
            output_size,
            unique_suh_ptrs,
            had_group_ids_tensor,
        )
        logger.debug(
            "EXL3 QKV MGEMM prepared %s: %d x K5120xN%d, %.1f MiB "
            "additional sliced weights.",
            prefix,
            expected_count,
            output_size,
            copied_bytes / 1024**2,
        )
        if _BF16_IO_ENABLED:
            logger.info(
                "EXL3 BF16-I/O bundle prepared %s: matrices=%d, "
                "unique_input_hadamards=%d, N=%d.",
                prefix,
                expected_count,
                len(unique_suh_chunks),
                output_size,
            )

    @staticmethod
    def _validate_marker(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"EXL3 {name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != expected:
            raise ValueError(
                f"Invalid EXL3 {name} sentinel 0x{value:08x}; expected 0x{expected:08x}"
            )

    @classmethod
    def _validate_loaded_tensors(cls, layer: torch.nn.Module) -> None:
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.dtype != torch.int16 or trellis.ndim != 3:
                raise ValueError("EXL3 trellis must be rank-3 int16")
            if trellis.shape[2] % 16 or not 1 <= trellis.shape[2] // 16 <= 8:
                raise ValueError(
                    f"Invalid EXL3 trellis bit width {trellis.shape[2]} / 16"
                )
            if suh.dtype != torch.float16 or suh.ndim != 1:
                raise ValueError("EXL3 suh must be rank-1 float16")
            if svh.dtype != torch.float16 or svh.ndim != 1:
                raise ValueError("EXL3 svh must be rank-1 float16")
            k = trellis.shape[0] * 16
            n = trellis.shape[1] * 16
            if suh.numel() != k or svh.numel() != n:
                raise ValueError(
                    "EXL3 dimensions disagree: "
                    f"trellis={tuple(trellis.shape)}, suh={suh.numel()}, "
                    f"svh={svh.numel()}"
                )
            if k % _HADAMARD_BLOCK or n % _HADAMARD_BLOCK:
                raise ValueError(
                    f"EXL3 kernel dimensions must be {_HADAMARD_BLOCK}-aligned, "
                    f"got K={k}, N={n}"
                )
            if shard_id in layer.mcg.exl3_tensors:
                cls._validate_marker(
                    layer.mcg.exl3_tensors[shard_id], _MCG_SENTINEL, "mcg"
                )
            if shard_id in layer.mul1.exl3_tensors:
                cls._validate_marker(
                    layer.mul1.exl3_tensors[shard_id], _MUL1_SENTINEL, "mul1"
                )

    @staticmethod
    def _slice_exl3_tensor(
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        size: int,
    ) -> torch.Tensor:
        if start % _HADAMARD_BLOCK or size % _HADAMARD_BLOCK:
            axis = "output" if dim == 1 else "input"
            raise ValueError(
                f"EXL3 TP {axis} slice must be {_HADAMARD_BLOCK}-aligned, "
                f"got start={start}, size={size}"
            )
        return tensor.narrow(dim, start // 16, size // 16).contiguous()

    @staticmethod
    def _output_shard_size(layer: torch.nn.Module, shard_id: ShardId) -> int:
        if shard_id is None:
            return layer.exl3_output_partition_sizes[0]
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return layer.exl3_output_partition_sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, tuple):
            return sum(layer.exl3_output_partition_sizes[idx] for idx in shard_id)
        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]
        return layer.exl3_output_partition_sizes[layer.exl3_shard_ids.index(shard_id)]

    @staticmethod
    def _qkv_output_start(
        layer: torch.nn.Module, shard_id: ShardId, shard_size: int
    ) -> int:
        if shard_id in ("k", "v"):
            shard_rank = layer.exl3_tp_rank // layer.num_kv_head_replicas
        else:
            shard_rank = layer.exl3_tp_rank
        return shard_rank * shard_size

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return
        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.exl3_input_size_per_partition
            size = layer.exl3_input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = (
                    layer.suh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        already_sharded = cls._expand_tuple_output_shards(layer)
        for shard_id in layer.exl3_shard_ids:
            if shard_id in already_sharded:
                continue
            size = cls._output_shard_size(layer, shard_id)
            start = cls._qkv_output_start(layer, shard_id, size)
            layer.svh.exl3_tensors[shard_id] = (
                layer.svh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
            )
            layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=size,
            )

    @classmethod
    def _expand_tuple_output_shards(cls, layer: torch.nn.Module) -> set[int]:
        tuples = [sid for sid in layer.exl3_shard_ids if isinstance(sid, tuple)]
        if not tuples:
            return set()

        expanded_ids: list[ShardId] = []
        component_ids: set[int] = set()
        for shard_id in layer.exl3_shard_ids:
            if isinstance(shard_id, tuple):
                expanded_ids.extend(shard_id)
                component_ids.update(shard_id)
            else:
                expanded_ids.append(shard_id)

        for tuple_id in tuples:
            full_offsets: dict[int, int] = {}
            offset = 0
            for idx in tuple_id:
                full_offsets[idx] = offset
                offset += layer.exl3_output_partition_sizes[idx] * layer.exl3_tp_size
            for idx in tuple_id:
                size = layer.exl3_output_partition_sizes[idx]
                start = full_offsets[idx] + layer.exl3_tp_rank * size
                layer.suh.exl3_tensors[idx] = layer.suh.exl3_tensors[tuple_id]
                layer.svh.exl3_tensors[idx] = (
                    layer.svh.exl3_tensors[tuple_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[idx] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[tuple_id],
                    dim=1,
                    start=start,
                    size=size,
                )
                layer.exl3_expected_codebooks[idx] = layer.exl3_expected_codebooks[
                    tuple_id
                ]
                for marker in ("mcg", "mul1"):
                    tensors = getattr(layer, marker).exl3_tensors
                    if tuple_id in tensors:
                        tensors[idx] = tensors[tuple_id]
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, attr).exl3_tensors.pop(tuple_id, None)
            layer.exl3_expected_codebooks.pop(tuple_id, None)

        layer.exl3_shard_ids = expanded_ids
        return component_ids

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[ShardId]:
        if len(output_partition_sizes) == 1:
            return [None]
        prefix = getattr(layer, "prefix", "")
        if isinstance(layer, QKVParallelLinear) and len(output_partition_sizes) == 3:
            return ["q", "k", "v"]
        if prefix.endswith("in_proj_qkvz"):
            return [(0, 1, 2), 3]
        return list(range(len(output_partition_sizes)))

    def _source_prefixes_for_layer(
        self, layer: torch.nn.Module, shard_ids: list[ShardId]
    ) -> list[str]:
        prefix = getattr(layer, "prefix", "")
        if len(shard_ids) == 1:
            return [prefix or "lm_head"]
        leaf = prefix.rsplit(".", 1)[-1]
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        sources = self.quant_config.packed_modules_mapping.get(leaf)
        if sources and len(sources) == len(shard_ids):
            return [f"{base}.{source}" if base else source for source in sources]
        raise ValueError(
            f"EXL3 does not know the source matrices for packed layer {prefix}; "
            "add it to the model's packed_modules_mapping."
        )

    @staticmethod
    def _apply_one(
        layer: torch.nn.Module, x: torch.Tensor, shard_id: ShardId
    ) -> torch.Tensor:
        # Multi-precision dispatch: direct calls (same pattern as FP6-only patch)
        # b12x _STATE_LOCK is patched to nullcontext, so dynamo can trace
        # through b12x's runtime_control functions without errors.
        # Draft-only FP4 lm_head: the MTP model's compute_logits sets
        # _use_fp4_draft around its logits call; verify never sets it, so the
        # target's sampling path stays K6-exact. Draft and verify are captured
        # as separate CUDA graphs, so the flag bakes correctly into each.
        trellis = layer.trellis.exl3_tensors[shard_id]
        has_mcg = shard_id in layer.mcg.exl3_tensors
        has_mul1 = shard_id in layer.mul1.exl3_tensors
        output = None
        if output is None:
            # B12X W4A16 wins prefill decisively (fidelity profile PP 1504 ->
            # 1967 when the K6 matrices use it) but loses decode to the fused
            # exl3_gemm kernel, which also drafts better: TG-essay 93.3 vs 90.0
            # and MTP acceptance 0.304 vs 0.281, measured on otherwise identical
            # configs.  Both paths are trellis-exact (agreement verified by
            # VLLM_EXL3_B12X_SELFTEST at m=4 and m=3072, cos >= 0.999999), so
            # routing by row count is free.  0 = always prefer B12X.
            use_b12x = _b12x_trellis_k6_supported(
                trellis,
                has_mcg=has_mcg,
                has_mul1=has_mul1,
            ) and (_B12X_MIN_M == 0 or x.shape[0] >= _B12X_MIN_M)
            if (
                use_b12x
                and _PREFILL_FUSED_RECONSTRUCT_MIN_M > 0
                and x.shape[0] >= _PREFILL_FUSED_RECONSTRUCT_MIN_M
                and hasattr(_load_exl3_ext(), "reconstruct_had_slice")
            ):
                use_b12x = False
            # NOTE (PR #318's warning applies): this is a Python-level branch.
            # Safe under shape-specialised CUDA graphs (m is fixed per captured
            # graph) and with compile=NONE; if torch.compile with dynamic shapes
            # is ever enabled, move this dispatch INSIDE the custom op or it
            # will bake one branch into the traced graph.
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if (
                use_b12x
                and _B12X_SELFTEST
                and x.shape[0] > 0
                and x.abs().amax().item() > 0
            ):
                # One-shot per (layer, shard, shape): run BOTH routes on the
                # real served tensors and report agreement.  The standalone
                # harness (tools/b12x-k5-selftest.py) compares checkpoint
                # tensors, which cannot see padding, merged-shard views or
                # loader transforms; this sees exactly what serving uses.
                # m is part of the key: prefill and decode take different
                # kernels inside b12x (the cooperative small-M path vs the
                # scratch-consuming one), so testing only the first shape seen
                # would leave the decode route unverified.
                key = (
                    getattr(layer, "prefix", ""),
                    shard_id,
                    int(x.shape[0]),
                    int(x.shape[1]),
                    int(trellis.shape[1]) * 16,
                    int(trellis.shape[2]),
                )
                if key not in _B12X_SELFTEST_DONE:
                    _B12X_SELFTEST_DONE.add(key)
                    got = _b12x_trellis_linear(x, trellis, suh, svh)
                    ref = _exl3_gemm(
                        x, trellis, suh, svh, has_mcg, has_mul1
                    )
                    a = got.float().flatten()
                    b = ref.float().flatten()
                    cos = torch.nn.functional.cosine_similarity(
                        a, b, dim=0
                    ).item()
                    rel = (
                        (a - b).abs().max() / b.abs().max().clamp_min(1e-6)
                    ).item()
                    logger.warning(
                        "B12X trellis selftest %s[%r] bits=%d m=%d K=%d "
                        "Npacked=%d suh=%d svh=%d: cos=%.6f max_rel=%.4g %s",
                        key[0], shard_id, int(trellis.shape[2]) // 16,
                        int(x.shape[0]), key[3], key[4],
                        int(suh.numel()), int(svh.numel()), cos, rel,
                        "OK" if (cos > 0.999 and rel < 0.05)
                        else "*** MISMATCH ***",
                    )
                    output = ref  # serve the reference on the selftest call
            if output is None:
                output = (
                    _b12x_trellis_linear(x, trellis, suh, svh)
                    if use_b12x
                    else _exl3_gemm(
                        x, trellis, suh, svh, has_mcg, has_mul1
                    )
                )
        logical_n = Exl3LinearMethod._output_shard_size(layer, shard_id)
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]
