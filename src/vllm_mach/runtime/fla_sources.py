# SPDX-License-Identifier: Apache-2.0
"""Pinned FLA source changes reviewed for the vLLM 0.34 runtime."""

from vllm_mach.stack import BASE_PACKAGES


_V034 = {
    "chunk_delta_h.py": "e900d52b8ab8e4479c21908176fd80b0ae6a4b8a194b213f02a029993d885d30",
    "fused_recurrent.py": "9915fb0fe69f3b1205eff1532cb51c4cb640badd4e1bdf8dddced89de6e2ed69",
    "l2norm.py": "5050521dfef0d66ed0539005ddb1808d2c6ddeb14e2faecc2927f785861b71da",
    "layernorm_guard.py": "014fe7c612f4d5987c23fdb8a2734eb415ef5de0cbee66309c13c95149ac992a",
    "solve_tril.py": "a3cb75fc5b20ef2b1222201cd91f350b5db6a82efa53c9b1a76ae2907ba4e81b",
    "utils.py": "0e2d474de9f15780d7d44eceb65673808f93177318a7e4ce7370eecf9f384a43",
}


def source_sha256(filename: str, v033_sha256: str) -> str:
    if BASE_PACKAGES["vllm"] == "0.34.0":
        return _V034.get(filename, v033_sha256)
    return v033_sha256
