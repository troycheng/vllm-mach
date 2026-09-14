# SPDX-License-Identifier: Apache-2.0
"""Rebuild the fixed token windows from an operator-supplied LongBench archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from common import json_sha, sha, write_json


def prepare(archive, tokenizer, output):
    from tokenizers import Tokenizer
    selection = json.loads(Path(__file__).with_name("calibration-selection.json").read_text())
    if sha(archive) != selection["archive_sha256"] or sha(tokenizer) != selection["tokenizer_sha256"]:
        raise ValueError("LongBench archive or tokenizer identity differs from the selected calibration")
    encoder = Tokenizer.from_file(str(tokenizer))
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as data:
        for kind in ("qa", "code"):
            group = selection[kind]
            files = {}
            samples = []
            for item in group["samples"]:
                name = item["source_file"]
                if name not in files:
                    raw = data.read(name)
                    if hashlib.sha256(raw).hexdigest() != item["source_file_sha256"]:
                        raise ValueError(f"Calibration source changed: {name}")
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    files[name] = {str(row["_id"]): row for row in rows}
                row = files[name][item["source_row_id"]]
                context = row["context"]
                if hashlib.sha256(context.encode()).hexdigest() != item["context_sha256"]:
                    raise ValueError("Calibration context changed")
                tokens = encoder.encode(context, add_special_tokens=False).ids
                start, p, n = item["window_start"], item["prompt_tokens"], item["target_tokens"]
                window = tokens[start:start + p + n]
                if len(window) != p + n or json_sha(window) != item["window_token_sha256"]:
                    raise ValueError("Calibration token window changed")
                samples.append({**item, "prompt_token_ids": window[:p], "target_token_ids": window[p:]})
            write_json(output / f"{kind}.json", {**group, "samples": samples,
                       "selection_sha256": sha(Path(__file__).with_name("calibration-selection.json"))})
