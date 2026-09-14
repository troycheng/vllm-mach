#!/usr/bin/env python3
"""Export a local model and content manifest for reproducible offline deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

SMALL_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(source: Path, destination: Path, origin: str, revision: str, hardlink=False):
    source, destination = source.resolve(), destination.resolve()
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    names = set(index["weight_map"].values()) | {index_path.name, "config.json"}
    names.update(name for name in SMALL_FILES if (source / name).is_file())
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe model filename: {name}")
        if not (source / name).is_file():
            raise ValueError(f"Missing model file: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in sorted(names):
        src, dst = source / name, destination / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            os.link(src.resolve(), dst)
        else:
            shutil.copyfile(src, dst)
        entries.append({"file": name, "bytes": dst.stat().st_size, "sha256": digest(dst)})
        print(f"verified {name}", flush=True)
    manifest = {"schema": "vllm-mach-model/v1", "source_model": origin, "source_revision": revision,
                "quantization": json.loads((destination / "config.json").read_text()).get("quantization_config"),
                "files": entries, "total_bytes": sum(e["bytes"] for e in entries)}
    (destination / "mach-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root):
    manifest = json.loads((root / "mach-manifest.json").read_text())
    if manifest.get("schema") != "vllm-mach-model/v1":
        raise ValueError("Unsupported model manifest")
    for entry in manifest["files"]:
        name = entry["file"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe manifest filename: {name}")
        path = root / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Model content mismatch: {name}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--source-model")
    p.add_argument("--source-revision")
    p.add_argument("--hardlink", action="store_true", help="Same-filesystem export; treat both directories as immutable")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        result = verify(args.model)
    else:
        if not args.output or not args.source_model or not args.source_revision:
            p.error("export requires --output, --source-model and --source-revision")
        result = export(args.model, args.output, args.source_model, args.source_revision, args.hardlink)
    print(json.dumps({"files": len(result["files"]), "bytes": result["total_bytes"]}))


if __name__ == "__main__":
    main()
