"""Freeze repeated ShareGPT token prompts, matching the local serving workload."""

import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--count", type=int, default=160)
    a = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    source = json.loads(a.dataset.read_text())
    rows = [
        r for r in source if len(r.get("conversations", r.get("conversation", []))) >= 2
    ]
    random.Random(a.seed).shuffle(rows)
    prompts = []
    for row in rows:
        text = row.get("conversations", row.get("conversation"))[0]["value"]
        tokens = tokenizer.encode(text)
        if tokens:
            prompts.append(dict(source_id=row.get("id"), token_ids=tokens[:3000]))
        if len(prompts) == a.count:
            break
    assert len(prompts) == a.count
    with a.output.open("x") as stream:
        json.dump(
            dict(
                source="ShareGPT_V3_unfiltered_cleaned_split.json",
                seed=a.seed,
                input_tokens=3000,
                prompts=prompts,
            ),
            stream,
            separators=(",", ":"),
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
