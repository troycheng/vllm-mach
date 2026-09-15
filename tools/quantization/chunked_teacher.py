# SPDX-License-Identifier: Apache-2.0
"""Bridge the frozen teacher hook across synchronized chunked-prefill waves.

Install the frozen ``teacher_decode_hook`` first. This bridge delegates the 23
negative-offset waves to the original native sampler. The deployed Sampler then
computes ``num_sampled=0`` for those unfinished prefills; the outer wrapper
records and checks that result. Offset zero and decode remain entirely under
the frozen teacher hook.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROWS = 32
PROMPT_LEN_SUBMITTED = 2999
CHUNK = 128
PARTIAL_WAVES = 23
EXPECTED_NEGATIVE_OFFSETS = tuple(
    CHUNK * wave - PROMPT_LEN_SUBMITTED for wave in range(1, PARTIAL_WAVES + 1)
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _teacher_rows(sampler, idx_mapping_np):
    table = getattr(sampler, "_k4o_teacher", {})
    active = [int(index) in table for index in idx_mapping_np]
    return table, active


def _offsets(sampler, idx_mapping_np, positions):
    table = sampler._k4o_teacher
    return [
        int(position) - int(table[int(index)]["prompt_len"]) + 1
        for index, position in zip(idx_mapping_np, positions, strict=True)
    ]


def _patch_sampler_class(Sampler):
    """Patch an already teacher-patched Sampler class; factored for CPU tests."""
    _require(
        hasattr(Sampler, "_k4o_teacher_originals"),
        "install frozen teacher_decode_hook before chunked bridge",
    )
    _require(
        not hasattr(Sampler, "_k4o_chunked_teacher_bridge_originals"),
        "chunked teacher bridge installed twice",
    )
    teacher_sample, teacher_call = Sampler.sample, Sampler.__call__
    native_sample = Sampler._k4o_teacher_originals[1]
    Sampler._k4o_chunked_teacher_bridge_originals = (teacher_sample, teacher_call)

    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        *,
        return_logprobs,
    ):
        table, active = _teacher_rows(self, idx_mapping_np)
        if not any(active):
            return teacher_sample(
                self,
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )
        _require(
            all(active) and len(active) == ROWS and int(logits.shape[0]) == ROWS,
            f"mixed/non-M32 teacher batch: active={active} logits={tuple(logits.shape)}",
        )
        import torch

        _require(
            not torch.cuda.is_current_stream_capturing(),
            "chunked teacher sampler must execute outside CUDA graph capture",
        )
        positions = pos.detach().cpu().tolist()
        offsets = _offsets(self, idx_mapping_np, positions)
        unique = set(offsets)
        _require(
            len(unique) == 1, f"ready/partial teacher rows mixed: offsets={offsets}"
        )
        offset = offsets[0]
        if offset >= 0:
            return teacher_sample(
                self,
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )
        _require(
            offset in EXPECTED_NEGATIVE_OFFSETS,
            f"unexpected partial-prefill offset {offset}",
        )
        _require(
            getattr(self, "_k4o_chunked_partial_pending", None) is None,
            "partial-prefill sample escaped its enclosing __call__",
        )
        self._k4o_chunked_partial_pending = {
            "offset": offset,
            "ids": [table[int(index)]["id"] for index in idx_mapping_np],
            "positions": [int(value) for value in positions],
        }
        # This produces an otherwise normal sampled tensor. The deployed outer
        # Sampler.__call__ subsequently zeroes num_sampled for unfinished chunks.
        return native_sample(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )

    def call(self, logits, input_batch):
        self._k4o_chunked_partial_pending = None
        result = teacher_call(self, logits, input_batch)
        pending = self._k4o_chunked_partial_pending
        self._k4o_chunked_partial_pending = None
        if pending is None:
            return result
        _require(
            int(input_batch.num_reqs) == ROWS and int(input_batch.num_tokens) == 4096,
            f"partial wave geometry changed: reqs={input_batch.num_reqs} tokens={input_batch.num_tokens}",
        )
        scheduled = [int(value) for value in input_batch.num_scheduled_tokens.tolist()]
        _require(
            scheduled == [CHUNK] * ROWS, f"partial wave is not 32x128: {scheduled}"
        )
        prefilling = [bool(value) for value in input_batch.is_prefilling_np.tolist()]
        _require(
            prefilling == [True] * ROWS,
            f"partial wave contains ready request: {prefilling}",
        )
        mapped = [
            self._k4o_teacher[int(index)]["id"] for index in input_batch.idx_mapping_np
        ]
        _require(
            mapped == pending["ids"] and len(set(mapped)) == ROWS,
            "partial-wave teacher request mapping changed or aliases",
        )
        num_sampled = [
            int(value) for value in result.num_sampled.detach().cpu().tolist()
        ]
        num_rejected = [
            int(value) for value in result.num_rejected.detach().cpu().tolist()
        ]
        _require(
            num_sampled == [0] * ROWS and num_rejected == [0] * ROWS,
            f"unfinished chunk emitted/rejected tokens: sampled={num_sampled} rejected={num_rejected}",
        )
        if not hasattr(self, "_k4o_chunked_partial_observations"):
            self._k4o_chunked_partial_observations = []
        self._k4o_chunked_partial_observations.append(
            {
                **pending,
                "num_tokens": int(input_batch.num_tokens),
                "num_reqs": int(input_batch.num_reqs),
                "num_scheduled_tokens": scheduled,
                "is_prefilling": prefilling,
                "num_sampled": num_sampled,
                "num_rejected": num_rejected,
            }
        )
        return result

    Sampler.sample = sample
    Sampler.__call__ = call


def install(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    sampler_path = Path(inspect.getfile(Sampler)).resolve()
    input_path = Path(inspect.getfile(InputBatch)).resolve()
    _patch_sampler_class(Sampler)
    sampler = worker.model_runner.sampler
    _require(
        isinstance(sampler, Sampler)
        and getattr(worker, "_k4o_teacher_sampler", None) is sampler,
        "frozen teacher hook is not installed on worker sampler",
    )
    sampler._k4o_chunked_partial_observations = []
    worker._k4o_chunked_teacher_sampler = sampler
    return {
        "rank": int(get_tensor_model_parallel_rank()),
        "protocol": "chunked_teacher_bridge_v1",
        "rows": ROWS,
        "partial_offsets": list(EXPECTED_NEGATIVE_OFFSETS),
        "sampler_source": str(sampler_path),
        "input_batch_source": str(input_path),
        "native_partial_sample": True,
        "teacher_offset_zero_and_decode_unchanged": True,
    }


def observations(worker):
    from vllm.distributed import get_tensor_model_parallel_rank

    sampler = worker._k4o_chunked_teacher_sampler
    rows = list(sampler._k4o_chunked_partial_observations)
    sampler._k4o_chunked_partial_observations.clear()
    offsets = [int(item["offset"]) for item in rows]
    _require(
        offsets == list(EXPECTED_NEGATIVE_OFFSETS),
        f"partial-prefill wave sequence mismatch: {offsets}",
    )
    expected_ids = set(rows[0]["ids"]) if rows else set()
    _require(
        len(expected_ids) == ROWS
        and all(set(item["ids"]) == expected_ids for item in rows),
        "partial-prefill request set changed",
    )
    return {
        "rank": int(get_tensor_model_parallel_rank()),
        "calls": rows,
        "partial_wave_count": len(rows),
        "offsets": offsets,
    }
