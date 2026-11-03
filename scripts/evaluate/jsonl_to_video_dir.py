#!/usr/bin/env python3
"""Materialize a persistent flat symlink directory from a video JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}


def find_dataset_root(path: Path) -> Path:
    for parent in (path.parent, *path.parent.parents):
        if (parent / "fake").is_dir() and (parent / "real").is_dir():
            return parent
    return path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Root for paths in file_name/rel_path; auto-detected for FakePartsBench JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", choices=("fake", "real", "all"), default="fake")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"JSONL input does not exist: {input_jsonl}")
    root = args.input_root.expanduser().resolve() if args.input_root else find_dataset_root(input_jsonl)
    records: list[tuple[Path, str]] = []
    with input_jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {input_jsonl}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {input_jsonl}:{line_number}")
            raw_label = str(row.get("label", "")).strip().lower()
            label = {"1": "fake", "true": "fake", "0": "real", "false": "real"}.get(raw_label, raw_label)
            if args.label != "all" and label and label != args.label:
                continue
            value = row.get("file_name") or row.get("rel_path") or row.get("path")
            if not value:
                raise ValueError(f"missing file_name, rel_path, or path at {input_jsonl}:{line_number}")
            source = Path(str(value)).expanduser()
            if not source.is_absolute():
                candidates = [root / source]
                if source.parts and source.parts[0] == root.name:
                    candidates.insert(0, root.parent / source)
                source = next((item for item in candidates if item.is_file()), candidates[0])
            source = source.resolve()
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"unsupported video type at {input_jsonl}:{line_number}: {source}")
            if not source.is_file():
                raise FileNotFoundError(f"video listed at {input_jsonl}:{line_number} does not exist: {source}")
            records.append((source, label))

    if not records:
        raise ValueError(f"no matching videos found in {input_jsonl} for label={args.label}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (source, _label) in enumerate(records):
        link = output_dir / f"{index:08d}{source.suffix.lower()}"
        if link.is_symlink():
            if link.resolve() != source:
                raise FileExistsError(f"existing input link points to a different video: {link}")
        elif link.exists():
            raise FileExistsError(f"refusing to replace existing input: {link}")
        else:
            link.symlink_to(source)
    print(len(records))


if __name__ == "__main__":
    main()
