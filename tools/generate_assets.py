#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate MXFP6 or NVFP4/rank64 assets with the existing quantizers."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "quantization"))
from common import checked_model, sha, write_json

CONVERTER = "37242a6a1bf6869857084d2ac7ccb22d1af7168d"
IGNORE = ["model.language_model.embed_tokens", "lm_head", "re:.*linear_attn.in_proj_a$",
          "re:.*linear_attn.in_proj_b$", "re:.*visual.*", "re:^mtp.*"]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="kind", required=True)
    for kind in ("mxfp6", "rank64"):
        arm = sub.add_parser(kind)
        arm.add_argument("--model", type=Path, required=True, help="Original BF16 checkpoint")
        arm.add_argument("--output", type=Path, required=True, help="New output directory")
        arm.add_argument("--device", default="cuda:0")
        arm.add_argument("--dry-run", action="store_true")
        if kind == "rank64":
            arm.add_argument("--exl3", type=Path, required=True)
            arm.add_argument("--mxfp6", type=Path, required=True)
            arm.add_argument("--calibration-archive", type=Path, required=True)
            arm.add_argument("--work-dir", type=Path, required=True, help="New directory for calibration and intermediate tensors")
    return p.parse_args(argv)


def rank_commands(args):
    work = args.work_dir.resolve()
    common = ["--model", str(args.exl3.resolve()), "--mxfp6", str(args.mxfp6.resolve()),
              "--tokenizer", str(args.model.resolve())]
    commands = [[sys.executable, str(HERE / "quantization/capture.py"), *common,
                 "--manifest", str(work / "manifests" / f"{kind}.json"),
                 "--output", str(work / kind)] for kind in ("qa", "code")]
    commands.append([sys.executable, str(HERE / "quantization/build_rank64.py"),
                     "--model", str(args.model.resolve()), "--work-dir", str(work),
                     "--output", str(args.output.resolve()), "--device", args.device])
    return commands


def main():
    args = arguments()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    checked_model(args.model)
    if args.output.exists() or args.output.is_relative_to(args.model) or args.model.is_relative_to(args.output):
        raise ValueError("Output must be a new directory separate from the source model")
    if args.kind == "mxfp6":
        recipe = dict(model_stub=str(args.model), save_directory=str(args.output), scheme="MXFP6",
                      ignore=IGNORE, max_workers=1, device=args.device)
        if args.dry_run:
            print(json.dumps(recipe, indent=2))
            return
        dist = importlib.metadata.distribution("llmcompressor")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != CONVERTER:
            raise RuntimeError(f"Install llm-compressor directly from git at {CONVERTER} in a separate environment")
        from llmcompressor import model_free_ptq
        model_free_ptq(**recipe)
        write_json(args.output / "mach-generation.json", {
            "recipe": recipe, "converter_commit": CONVERTER,
            "source_config_sha256": sha(args.model / "config.json"),
            "source_index_sha256": sha(args.model / "model.safetensors.index.json")})
        return
    commands = rank_commands(args)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    work = args.work_dir.resolve()
    for source in (args.model, args.exl3.resolve(), args.mxfp6.resolve(), args.output):
        if work == source or work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("Work, output and model directories must be separate")
    for source in (args.exl3, args.mxfp6):
        if args.output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.output):
            raise ValueError("Output and input model directories must be separate")
        if not (source / "config.json").is_file() or not (source / "model.safetensors.index.json").is_file():
            raise ValueError(f"Missing model config/index: {source}")
    work.mkdir(parents=True, exist_ok=False)
    from prepare_calibration import prepare
    prepare(args.calibration_archive, args.model / "tokenizer.json", work / "manifests")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
