"""Collect the matched physical-M4 persistent diagnostic, separate from M32."""

import argparse
import json
from pathlib import Path

import numpy as np

from collect_native_comparison import bootstrap


def collect(root):
    def read(arm, name):
        return json.loads((root / arm / name).read_text())

    reference = read("bf16", "records.json")
    ids = [r["id"] for r in reference]
    assert len(ids) == len(set(ids)) == 256
    assert sum(len(r["gold_logprobs"]) for r in reference) == 10479
    runs = {}
    for arm in ("default", "persistent", "gdn", "full_ba", "full_gdn"):
        records = read(arm, "records.json")
        assert [r["id"] for r in records] == ids
        contract = read(arm, "contract.json")
        assert contract["physical_rows"] == 4
        errors = []
        for a, b in zip(reference, records, strict=True):
            assert len(a["gold_logprobs"]) == len(b["gold_logprobs"])
            delta = np.asarray(a["gold_logprobs"]) - b["gold_logprobs"]
            assert np.isfinite(delta).all()
            errors.append(float(np.abs(delta).mean()))
        runs[arm] = dict(
            mae=float(np.mean(errors)),
            ci95=bootstrap(errors),
            query_mae=errors,
            gold_logprobs=[r["gold_logprobs"] for r in records],
            contract=contract,
            repeat=read(arm, "repeat.json"),
            dispatch=read(arm, "gdn.json"),
        )
    assert all(
        r["prepared_layers"] == 48 and r["persistent_m4"] > 0
        for r in runs["persistent"]["dispatch"]
    )
    assert runs["gdn"]["gold_logprobs"] == runs["persistent"]["gold_logprobs"]
    paired = np.asarray(runs["persistent"]["query_mae"]) - runs["default"]["query_mae"]
    return dict(
        schema="native-gdn-m4-fidelity/v1",
        physical_rows=4,
        query_count=256,
        target_tokens=10479,
        reference_contract=read("bf16", "contract.json"),
        reference_repeat=read("bf16", "repeat.json"),
        queries=[
            dict(id=r["id"], domain=r["domain"], bf16_logprobs=r["gold_logprobs"])
            for r in reference
        ],
        runs=runs,
        full_persistent_minus_native=dict(
            mean=float(
                np.mean(
                    np.asarray(runs["full_gdn"]["query_mae"])
                    - runs["full_ba"]["query_mae"]
                )
            ),
            ci95=bootstrap(
                np.asarray(runs["full_gdn"]["query_mae"]) - runs["full_ba"]["query_mae"]
            ),
        ),
        persistent_minus_default=dict(
            mean=float(paired.mean()), ci95=bootstrap(paired)
        ),
        bootstrap=dict(seed=20260910, replicates=20000, unit="query"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    result = collect(a.results)
    Path(__file__).with_name("gdn-m4-fidelity.json").write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print({k: (v["mae"], v["ci95"]) for k, v in result["runs"].items()})
    print(result["persistent_minus_default"])


if __name__ == "__main__":
    main()
