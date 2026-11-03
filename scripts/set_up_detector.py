#!/usr/bin/env python3
"""Prepare checkpoints for explicitly selected detector methods/versions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fakeparts.detection.config import DET_MODELS, canonical_detector_id, resolve_checkpoints
from fakeparts.detection.utils import ensure_checkpoints


def checkpoint_is_present(checkpoint) -> bool:
    return checkpoint.path.is_file() or (
        checkpoint.path.is_dir()
        and bool(checkpoint.required_files)
        and all((checkpoint.path / item).is_file() for item in checkpoint.required_files)
    )


def has_setup_source(checkpoint) -> bool:
    return bool(checkpoint.url or checkpoint.hf_repo or checkpoint.gdrive_id)


def detector_methods_help() -> str:
    grouped: dict[str, list[str]] = {}
    for name, config in sorted(DET_MODELS.items()):
        label = config.family
        versions = sorted(config.checkpoint_variants)
        method_ids = [f"{name}:{version}" for version in versions] if versions else [name]
        grouped.setdefault(label, []).extend(method_ids)
    return (
        "method IDs to prepare by detector category:\n"
        + "\n".join(
            f"  {label}:\n" + "\n".join(f"    {method}" for method in methods)
            for label, methods in grouped.items()
        )
        + "\n  Use METHOD:VERSION for a specific checkpoint version."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--methods", action="extend", nargs="+", default=None, metavar="METHOD[:VERSION]",
                        help=detector_methods_help() + "\n  Defaults to methods with complete checkpoint sources.")
    parser.add_argument("--dry-run", action="store_true", help="print selected paths and sources only")
    parser.add_argument("--overwrite", action="store_true", help="force-refresh selected remote checkpoints")
    args = parser.parse_args()

    if args.methods is None:
        args.methods = []
        skipped = []
        for name, config in sorted(DET_MODELS.items()):
            try:
                version, checkpoints = resolve_checkpoints(config)
            except ValueError:
                skipped.append(name)
                continue
            if checkpoints and all(
                checkpoint_is_present(item) or has_setup_source(item) for item in checkpoints
            ):
                args.methods.append(name)
            else:
                skipped.append(name)
        print("--methods omitted; selected methods with a complete local checkpoint set or registered download sources.", flush=True)
        if skipped:
            print("Skipped methods without a complete setup source: " + ", ".join(skipped), flush=True)

    selected = []
    seen = set()
    for token in args.methods:
        for raw in token.split(","):
            raw_model, separator, raw_version = raw.partition(":")
            try:
                model = canonical_detector_id(raw_model)
            except KeyError:
                parser.error(f"unknown detector {raw!r}; registered: {', '.join(sorted(DET_MODELS))}")
            try:
                version, checkpoints = resolve_checkpoints(
                    DET_MODELS[model], raw_version.strip() if separator else None)
            except ValueError as error:
                parser.error(str(error))
            key = (model, version)
            if key not in seen:
                selected.append((model, version, checkpoints))
                seen.add(key)
    if not selected:
        parser.error("--methods must contain at least one method ID")

    checkpoint_set = {}
    for model, version, checkpoints in selected:
        print(f"[{model}:{version}]", flush=True)
        if not DET_MODELS[model].implemented:
            print("  Adapter is not implemented; preparing its registered checkpoint set only.", flush=True)
        if not checkpoints:
            print("  No checkpoint registered.", flush=True)
            continue
        for checkpoint in checkpoints:
            checkpoint_set.setdefault(checkpoint.path, checkpoint)
            if checkpoint.hf_repo:
                suffix = f"/{checkpoint.hf_filename}" if checkpoint.hf_filename else " (repository snapshot)"
                source = f"HF:{checkpoint.hf_repo}{suffix} via huggingface_hub Python API"
            elif checkpoint.gdrive_id:
                source = f"Google Drive:{checkpoint.gdrive_id}"
            else:
                source = checkpoint.url or "manual/local"
            valid = checkpoint_is_present(checkpoint)
            state = "present" if valid else "missing"
            print(f"  {state}: {checkpoint.path}  [{source}]", flush=True)
        if args.dry_run:
            continue
    if args.dry_run or not checkpoint_set:
        return 0
    try:
        ensure_checkpoints(
            tuple(checkpoint_set.values()),
            download=True,
            force_download=args.overwrite,
        )
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"Setup incomplete: {error}", file=sys.stderr, flush=True)
        return 1
    print(f"Checkpoint set ready ({len(checkpoint_set)} unique files).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
