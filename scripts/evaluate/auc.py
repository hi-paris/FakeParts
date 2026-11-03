#!/usr/bin/env python3
"""Balance detector score CSVs and update per-detector and category AUC tables."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import gzip
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import fmean, median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fakeparts.detection.config import (  # noqa: E402
    DET_MODELS, REAL_REFERENCE_CLASSES, canonical_detector_id, resolve_checkpoints,
)
from fakeparts.naming import (
    DETECTOR_ORDER, FAMILY_ORDER, GENERATORS, GENERATOR_ORDER, TASKS, TASK_ORDER,
    canonical_generator_id, canonical_task_id, table_order_key, version_name,
)

METHOD_FIELDS = (
    "detector_family_full_name", "detector_family_short_name", "detector_family_id",
    "detector_full_name", "detector_short_name", "detector_id",
    "detector_version_name", "detector_version_id",
    "generator_task_full_name", "generator_task_short_name", "generator_task_id",
    "generator_full_name", "generator_short_name", "generator_id", "generator_category_id", "fake_videos", "real_videos", "unique_real_videos",
    "seed", "roc_auc_percent",
)
CATEGORY_FIELDS = (
    "detector_family_full_name", "detector_family_short_name", "detector_family_id",
    "detector_full_name", "detector_short_name", "detector_id",
    "detector_version_name", "detector_version_id",
    "generator_task_full_name", "generator_task_short_name", "generator_task_id",
    "generator_task_methods", "fake_videos", "real_videos",
    "macro_roc_auc_percent",
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
FAMILY_ORDER_INDEX = {name: index for index, name in enumerate(FAMILY_ORDER)}
DETECTOR_ORDER_INDEX = {name: index for index, name in enumerate(DETECTOR_ORDER)}

GENERATOR_SOURCES = {
    "sora_1080p": "Report'24", "sora_720p": "Report'24",
    "opensora": "ArXiv'24", "cogvidx": "ICLR'25",
    "allegroai": "ArXiv'24", "mochi1": "Report'24",
    "ltx_video": "ArXiv'24", "hunyuanvid": "ArXiv'24",
    "veo2": "Report'24", "wan21": "ArXiv'25",
    "rave": "CVPR'24", "anyv2v": "TMLR'24",
    "insightface": "CVPR'19", "liveportrait": "ECCV'24",
    "hunyuanportrait": "ArXiv'25",
    "propainter": "ICCV'23", "rovi": "CVPR'24",
    "diffueraser": "ArXiv'25", "akira": "CVPR'25",
    "framer": "ICLR'25", "cosmos2": "ArXiv'25",
}


def normalize(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def key_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    marker = "/FakePartsBench/"
    if marker.lower() in text.lower():
        text = text[text.lower().rfind(marker.lower()) + len(marker):]
    elif text.lower().startswith("fakepartsbench/"):
        text = text[len("FakePartsBench/"):]
    text = re.sub(r"^\./+", "", text)
    return str(PurePosixPath(text))


def path_metadata(key: str) -> tuple[str, str, str]:
    parts = PurePosixPath(key).parts
    if not parts:
        return "", "", ""
    if parts[0].lower() == "real":
        return "real", normalize(parts[1]) if len(parts) > 1 else "real", ""
    if parts[0].lower() == "fake" and len(parts) >= 4:
        return "fake", normalize(parts[1]), normalize(parts[2])
    return "", "", ""


def parse_label(value: object, key: str) -> str:
    label = str(value or "").strip().lower()
    if label in {"real", "0", "false"}:
        return "real"
    if label in {"fake", "1", "true"}:
        return "fake"
    return path_metadata(key)[0]


def rank_auc(scores: list[tuple[float, str]]) -> float:
    ordered = sorted(scores, key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label == "fake" for _, label in ordered[index:end])
        index = end
    positives = sum(label == "fake" for _, label in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires at least one fake and one real video")
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def is_result_csv(path: Path) -> bool:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream), [])
    return "score" in fields and bool({"rel_path", "file_name", "video"} & set(fields))


def read_video_scores(paths: list[Path]) -> dict[str, dict[str, object]]:
    collected: dict[str, dict[str, object]] = {}
    for path in paths:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "score" not in reader.fieldnames:
                raise ValueError(f"{path} has no score column")
            for row_no, row in enumerate(reader, 2):
                key = key_path(row.get("rel_path") or row.get("file_name") or row.get("video"))
                if not key:
                    continue
                try:
                    score = float(row.get("score", ""))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                label = parse_label(row.get("label"), key)
                path_label, path_task, path_method = path_metadata(key)
                task = normalize(row.get("class") or row.get("task") or row.get("class_name") or path_task)
                method = normalize(row.get("method") or path_method)
                if path_label and label and label != path_label:
                    raise ValueError(
                        f"{path}:{row_no}: label {label!r} conflicts with path label "
                        f"{path_label!r} for {key}"
                    )
                if label == "fake" and path_task and task and task != path_task:
                    raise ValueError(
                        f"{path}:{row_no}: task {task!r} conflicts with path task "
                        f"{path_task!r} for {key}"
                    )
                if label == "fake" and path_method and method and method != path_method:
                    raise ValueError(
                        f"{path}:{row_no}: method {method!r} conflicts with path method "
                        f"{path_method!r} for {key}"
                    )
                if label == "real" and task == "real":
                    task = path_task
                item = collected.setdefault(key, {
                    "score_sum": 0.0, "score_count": 0,
                    "label": label, "task": task, "method": method,
                })
                if item["label"] and label and item["label"] != label:
                    raise ValueError(f"{path}:{row_no}: conflicting labels for {key}")
                item["label"] = item["label"] or label or path_label
                if item["task"] and task and item["task"] != task:
                    raise ValueError(f"{path}:{row_no}: conflicting generator tasks for {key}")
                if item["method"] and method and item["method"] != method:
                    raise ValueError(f"{path}:{row_no}: conflicting generator methods for {key}")
                item["task"] = item["task"] or task
                item["method"] = item["method"] or method
                item["score_sum"] = float(item["score_sum"]) + score
                item["score_count"] = int(item["score_count"]) + 1
    return {
        key: {**item, "score": float(item["score_sum"]) / int(item["score_count"])}
        for key, item in collected.items() if item["score_count"]
    }


def choose_reals(count: int, real_keys: list[str], rng: random.Random) -> tuple[list[str], int]:
    if not real_keys:
        raise ValueError("No scored rows found in real_*.csv")
    order = list(real_keys)
    rng.shuffle(order)
    selected = order[:count]
    while len(selected) < count:
        order = list(real_keys)
        rng.shuffle(order)
        selected.extend(order[:min(len(order), count - len(selected))])
    return selected, len(set(selected))


def atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_grid_csv(path: Path, rows: list[list[object]]) -> None:
    """Write a presentation CSV whose header may span multiple rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    temporary.replace(path)


def version_match(path: Path, version: str, has_variants: bool) -> bool:
    name = path.name.lower()
    if has_variants or version != "default":
        tokens = [token for token in re.split(r"[^a-z0-9]+", version.lower()) if token]
        suffix_pattern = r"[._-]+".join(re.escape(token) for token in tokens)
        return bool(re.search(rf"[._-]{suffix_pattern}\.csv(?:\.gz)?$", name))
    # Accept pre-versioned default outputs while new detection runs emit
    # the explicit ``.default.csv`` suffix.
    if name.endswith(".default.csv") or name.endswith(".default.csv.gz"):
        return True
    if name.endswith(".csv.gz"):
        stem = name[:-7]
    elif name.endswith(".csv"):
        stem = name[:-4]
    else:
        return False
    return "." not in stem


def detector_summary(detector: str, version: str, result_dir: Path, seed: int) -> Path:
    try:
        detector = canonical_detector_id(detector)
    except KeyError:
        raise ValueError(f"Unknown detector {detector!r}; choices: {', '.join(sorted(DET_MODELS))}")
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Detector results directory does not exist: {result_dir}")
    config = DET_MODELS[detector]
    files = sorted(path for path in result_dir.iterdir()
                   if path.is_file() and version_match(path, version, bool(config.checkpoint_variants))
                   and is_result_csv(path))
    if not files:
        raise FileNotFoundError(
            f"No score CSVs for detector version {version!r} in {result_dir}"
        )

    all_scores = read_video_scores(files)
    fake_scores = {
        key: row for key, row in all_scores.items()
        if row["label"] == "fake"
    }
    if not fake_scores:
        raise FileNotFoundError(f"No scored fake CSVs found in {result_dir}")

    expected_real_classes = {normalize(name) for name in REAL_REFERENCE_CLASSES}
    available_real_classes = {
        str(row["task"]) for row in all_scores.values()
        if row["label"] in {"", "real"} and str(row["task"]) in expected_real_classes
    }
    missing_real_classes = sorted(expected_real_classes - available_real_classes)
    if missing_real_classes:
        raise FileNotFoundError(
            "AUC needs scored real videos for configured FakePartsBench/real classes: "
            + ", ".join(missing_real_classes)
            + f"; searched version-matched score CSVs in {result_dir}"
        )
    real_keys = sorted(
        key for key, row in all_scores.items()
        if row["label"] in {"", "real"} and str(row["task"]) in expected_real_classes
    )
    if not real_keys:
        raise ValueError(f"No valid configured real-video scores found in {result_dir}")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in fake_scores.items():
        if row["label"] != "fake":
            continue
        task, method = canonical_task_id(str(row["task"])), canonical_generator_id(str(row["method"]))
        if not task or not method:
            raise ValueError(f"Cannot determine task/method for fake video {key} in {result_dir}")
        groups[(task, method)].append(key)
    if not groups:
        raise ValueError(f"No valid fake-video scores found in {result_dir}")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for (task, method), fake_keys in sorted(groups.items()):
        selected_reals, unique_reals = choose_reals(len(fake_keys), real_keys, rng)
        fake_video_scores = [float(fake_scores[key]["score"]) for key in fake_keys]
        real_video_scores = [float(all_scores[key]["score"]) for key in selected_reals]
        paired_scores = [(score, "fake") for score in fake_video_scores]
        paired_scores.extend((score, "real") for score in real_video_scores)
        auc = rank_auc(paired_scores)
        if auc <= 0.05 or auc >= 0.95:
            print(
                f"[AUC sanity warning] {detector} × {task}/{method}: "
                f"AUC={auc * 100:.4f}%, reversed-score AUC={(1 - auc) * 100:.4f}%; "
                f"median fake score={median(fake_video_scores):.6g}, "
                f"median selected-real score={median(real_video_scores):.6g}. "
                "Check label and score direction; result is not auto-flipped."
            )
        family_full, family_short = config.family_full_name, config.family_short_name
        detector_full, detector_short = config.full_name, config.short_name
        task_full, task_short = TASKS.get(task, (task, task))
        generator_full, generator_short = GENERATORS.get(method, (method, method))
        rows.append({
            "detector_family_full_name": family_full,
            "detector_family_short_name": family_short,
            "detector_family_id": config.family,
            "detector_full_name": detector_full,
            "detector_short_name": detector_short,
            "detector_id": detector,
            "detector_version_name": version_name(detector, version),
            "detector_version_id": version,
            "generator_task_full_name": task_full,
            "generator_task_short_name": task_short,
            "generator_task_id": task,
            "generator_full_name": generator_full,
            "generator_short_name": generator_short,
            "generator_id": method,
            "generator_category_id": task,
            "fake_videos": len(fake_keys),
            "real_videos": len(fake_keys),
            "unique_real_videos": unique_reals,
            "seed": seed,
            "roc_auc_percent": round(auc * 100, 6),
        })
        print(f"{detector} × {task}/{method}: {auc * 100:.4f}% ({len(fake_keys)} fake + {len(fake_keys)} real)")

    summary = result_dir.parent / f"{detector}.csv"
    rows.sort(key=lambda row: table_order_key(
        str(row["generator_task_id"]), str(row["generator_id"]),
    ))
    previous: list[dict[str, str]] = []
    if summary.is_file():
        with summary.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) == METHOD_FIELDS:
                previous = list(reader)
            else:
                legacy_fields = ("detector", "detector_version", "detector_category", "generator_task", "generator_method", "generator_category", "fake_videos", "real_videos", "unique_real_videos", "seed", "roc_auc_percent")
                if tuple(reader.fieldnames or ()) != legacy_fields:
                    raise ValueError(f"Unexpected per-detector summary columns in {summary}")
                for old in reader:
                    config_old = DET_MODELS[old["detector"]]
                    task_old = canonical_task_id(old["generator_task"]); gen_old = canonical_generator_id(old["generator_method"])
                    tf, ts = TASKS.get(task_old, (task_old, task_old)); gf, gs = GENERATORS.get(gen_old, (gen_old, gen_old))
                    previous.append({"detector_family_full_name": config_old.family_full_name, "detector_family_short_name": config_old.family_short_name, "detector_family_id": config_old.family, "detector_full_name": config_old.full_name, "detector_short_name": config_old.short_name, "detector_id": old["detector"], "detector_version_name": version_name(old["detector"], old["detector_version"]), "detector_version_id": old["detector_version"], "generator_task_full_name": tf, "generator_task_short_name": ts, "generator_task_id": task_old, "generator_full_name": gf, "generator_short_name": gs, "generator_id": gen_old, "generator_category_id": task_old, "fake_videos": old["fake_videos"], "real_videos": old["real_videos"], "unique_real_videos": old["unique_real_videos"], "seed": old["seed"], "roc_auc_percent": old["roc_auc_percent"]})
    preserved = [row for row in previous if row["detector_version_id"] != version]
    merged = preserved + rows
    merged.sort(key=lambda row: (
        *table_order_key(str(row["generator_task_id"]), str(row["generator_id"])),
        str(row["detector_version_id"]),
    ))
    atomic_csv(summary, METHOD_FIELDS, merged)
    print(f"Updated: {summary}")
    return summary


def _recompute_detector_summaries(directory: Path, seed: int) -> str:
    """Compute all score-file versions for one detector in an isolated process."""
    detector = canonical_detector_id(directory.name)
    config = DET_MODELS[detector]
    versions: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file() or not is_result_csv(path):
            continue
        if config.checkpoint_variants:
            versions.update(
                version for version in config.checkpoint_variants
                if version_match(path, version, True)
            )
        elif version_match(path, "default", False):
            versions.add("default")

    for version in sorted(versions):
        try:
            detector_summary(detector, version, directory, seed)
        except FileNotFoundError as error:
            print(f"[skip] {detector}:{version}: {error}")
    return detector


def _auc_worker_count(task_count: int) -> int:
    cpu_slots = int(os.environ.get("SLURM_CPUS_PER_TASK", "") or os.cpu_count() or 1)
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "") or 0)
    if not mem_mb and os.environ.get("SLURM_MEM_PER_CPU"):
        mem_mb = int(os.environ["SLURM_MEM_PER_CPU"]) * cpu_slots
    # One detector process peaked at about 13.5 GiB in the recent full run.
    # Keep one GiB of headroom per process when Slurm provides a memory limit.
    memory_slots = max(1, mem_mb // (14 * 1024)) if mem_mb else cpu_slots
    return max(1, min(task_count, cpu_slots, memory_slots))


def summarize_all(paths: list[Path], seed: int = 0, recompute: bool = True) -> None:
    summaries: list[dict[str, str]] = []
    result_roots: set[Path] = set()
    seen: set[Path] = set()
    detector_dirs: list[Path] = []
    for input_path in paths:
        directory = input_path.expanduser().resolve()
        if not directory.is_dir():
            print(f"[skip] results path does not exist or is not a directory: {directory}")
            continue
        config = DET_MODELS.get(directory.name)
        if config is not None and directory.parent.name == config.family:
            detector_dirs.append(directory)
            continue

        # A family directory such as `results/oob/` contains detector
        # directories directly; a results root contains family directories.
        found = []
        family_dirs = [directory] if directory.name in FAMILY_ORDER else [
            child for child in directory.iterdir()
            if child.is_dir() and child.name in FAMILY_ORDER
        ]
        for family_dir in family_dirs:
            for detector_dir in family_dir.iterdir():
                config = DET_MODELS.get(detector_dir.name)
                if detector_dir.is_dir() and config is not None and config.family == family_dir.name:
                    found.append(detector_dir)
        if not found:
            print(f"[skip] no registered detector directories under: {directory}")
            continue
        detector_dirs.extend(found)

    detector_dirs = list(dict.fromkeys(detector_dirs))
    if not detector_dirs:
        raise FileNotFoundError("No registered detector directories found in the --all inputs")

    if recompute:
        workers = _auc_worker_count(len(detector_dirs))
        print(f"[parallel] Computing {len(detector_dirs)} detector summaries with {workers} Python worker(s)")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(
                _recompute_detector_summaries,
                detector_dirs,
                [seed] * len(detector_dirs),
            ))
    else:
        print(f"[aggregate] Reusing existing per-detector AUC CSVs for {len(detector_dirs)} detectors")

    for directory in detector_dirs:
        result_roots.add(directory.parent.parent)
        detector = canonical_detector_id(directory.name)
        config = DET_MODELS[detector]
        summary = directory.parent / f"{detector}.csv"
        if summary in seen:
            continue
        if not summary.is_file():
            print(f"[skip] no per-detector AUC summary: {summary}")
            continue
        seen.add(summary)
        with summary.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) == METHOD_FIELDS:
                summaries.extend(reader)
            elif tuple(reader.fieldnames or ()) == ("detector", "detector_version", "detector_category", "generator_task", "generator_method", "generator_category", "fake_videos", "real_videos", "unique_real_videos", "seed", "roc_auc_percent"):
                for old in reader:
                    config = DET_MODELS[old["detector"]]
                    task = canonical_task_id(old["generator_task"]); method = canonical_generator_id(old["generator_method"])
                    task_full, task_short = TASKS.get(task, (task, task)); gen_full, gen_short = GENERATORS.get(method, (method, method))
                    summaries.append({**old, "detector_family_full_name": config.family_full_name, "detector_family_short_name": config.family_short_name, "detector_family_id": config.family, "detector_full_name": config.full_name, "detector_short_name": config.short_name, "detector_id": old["detector"], "detector_version_name": version_name(old["detector"], old["detector_version"]), "detector_version_id": old["detector_version"], "generator_task_full_name": task_full, "generator_task_short_name": task_short, "generator_task_id": task, "generator_full_name": gen_full, "generator_short_name": gen_short, "generator_id": method, "generator_category_id": task, "detector": old["detector"], "detector_version": old["detector_version"], "detector_category": config.family.upper(), "generator_task": task, "generator_method": method, "generator_category": task})
            else:
                raise ValueError(f"Unexpected per-detector AUC columns in {summary}")
    if len(result_roots) != 1:
        raise ValueError("--all inputs must be detector directories under one common results/ root")
    if not summaries:
        raise FileNotFoundError("No per-detector AUC CSVs were found; run --method DETECTOR first")

    summaries.sort(key=lambda row: (
        FAMILY_ORDER_INDEX.get(row["detector_family_id"], len(FAMILY_ORDER_INDEX)),
        DETECTOR_ORDER_INDEX.get(row["detector_id"], len(DETECTOR_ORDER_INDEX)),
        row["detector_version_id"],
        *table_order_key(row["generator_task_id"], row["generator_id"]),
    ))
    root = next(iter(result_roots))
    auc_path = root / "auc.csv"
    detector_versions = [
        (detector, DET_MODELS[detector].default_variant or "default")
        for detector in DETECTOR_ORDER
    ]
    report_summaries = [
        row for row in summaries
        if row["detector_version_id"]
        == (DET_MODELS[row["detector_id"]].default_variant or "default")
    ]
    by_generator: dict[tuple[str, str], dict[str, object]] = defaultdict(dict)
    sample_counts: dict[tuple[str, str], tuple[int, int]] = {}
    for row in report_summaries:
        key = (row["generator_task_id"], row["generator_id"])
        detector = row["detector_id"]
        if detector in by_generator[key]:
            raise ValueError(f"Duplicate default AUC result for {key} and {detector}")
        by_generator[key][detector] = row["roc_auc_percent"]
        previous = sample_counts.get(key, (0, 0))
        sample_counts[key] = (
            max(previous[0], int(row["fake_videos"])),
            max(previous[1], int(row["real_videos"])),
        )

    auc_grid = [
        ["", "", "", "", ""]
        + [DET_MODELS[d].family_short_name for d, _ in detector_versions],
        ["", "", "", "", ""]
        + [DET_MODELS[d].short_name for d, _ in detector_versions],
        ["Gen Task", "Gen", "Source", "Fake Videos", "Real Videos"]
        + [version for _, version in detector_versions],
    ]
    for task in TASK_ORDER:
        if task == "real":
            continue
        for generator in GENERATOR_ORDER.get(task, ()):
            fake_count, real_count = sample_counts.get((task, generator), (0, 0))
            values = []
            for detector, _ in detector_versions:
                value = by_generator.get((task, generator), {}).get(detector, "")
                values.append("" if value == "" else f"{float(value):.2f}")
            auc_grid.append([
                TASKS.get(task, (task, task))[1],
                GENERATORS.get(generator, (generator, generator))[1],
                GENERATOR_SOURCES.get(generator, ""),
                fake_count, real_count, *values,
            ])
    atomic_grid_csv(auc_path, auc_grid)

    # Keep detector and checkpoint separate. Grouping only by detector family
    # mixes unrelated methods and cannot populate the per-detector task table.
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summaries:
        key = (
            row["detector_family_id"], row["detector_id"],
            row["detector_version_id"], row["generator_task_id"],
        )
        groups[key].append(row)
    category_rows = []
    for (detector_category, detector, version, task), group in sorted(
        groups.items(),
        key=lambda entry: (
            FAMILY_ORDER_INDEX.get(entry[0][0], len(FAMILY_ORDER_INDEX)),
            DETECTOR_ORDER_INDEX.get(entry[0][1], len(DETECTOR_ORDER_INDEX)),
            *table_order_key(entry[0][3]), entry[0][2],
        ),
    ):
        category_rows.append({
            "detector_family_full_name": DET_MODELS[detector].family_full_name,
            "detector_family_short_name": DET_MODELS[detector].family_short_name,
            "detector_family_id": detector_category.lower(),
            "detector_full_name": DET_MODELS[detector].full_name,
            "detector_short_name": DET_MODELS[detector].short_name,
            "detector_id": detector,
            "detector_version_name": version_name(detector, version),
            "detector_version_id": version,
            "generator_task_full_name": TASKS.get(task, (task, task))[0],
            "generator_task_short_name": TASKS.get(task, (task, task))[1],
            "generator_task_id": task,
            "generator_task_methods": len(group),
            "fake_videos": sum(int(row["fake_videos"]) for row in group),
            "real_videos": sum(int(row["real_videos"]) for row in group),
            "macro_roc_auc_percent": round(fmean(float(row["roc_auc_percent"]) for row in group), 6),
        })
    category_path = root / "auc_cate.csv"
    task_ids = [task for task in TASK_ORDER if task != "real"]
    task_short_names = [TASKS.get(task, (task, task))[1] for task in task_ids]
    reference_detector = "cnndet" if any(r["detector_id"] == "cnndet" for r in category_rows) else category_rows[0]["detector_id"]
    reference_rows = {r["generator_task_id"]: r for r in category_rows if r["detector_id"] == reference_detector}
    category_grid = [
        ["", "Gen Task", *task_short_names],
        ["", "Gen Meth", *[reference_rows.get(t, {}).get("generator_task_methods", "") for t in task_ids]],
        ["", "Fake Videos", *[reference_rows.get(t, {}).get("fake_videos", "") for t in task_ids]],
        ["", "Real Videos", *[reference_rows.get(t, {}).get("real_videos", "") for t in task_ids]],
    ]
    seen_detector_versions = {(r["detector_id"], r["detector_version_id"]) for r in category_rows}
    for detector, version in sorted(seen_detector_versions, key=lambda pair: (
        FAMILY_ORDER_INDEX.get(DET_MODELS[pair[0]].family, len(FAMILY_ORDER_INDEX)),
        DETECTOR_ORDER_INDEX.get(pair[0], len(DETECTOR_ORDER_INDEX)), pair[1],
    )):
        model = DET_MODELS[detector]
        versions = {v for d, v in seen_detector_versions if d == detector}
        detector_label = model.short_name
        if len(versions) > 1:
            detector_label = f"{detector_label} ({version_name(detector, version)})"
        lookup = {r["generator_task_id"]: r["macro_roc_auc_percent"] for r in category_rows
                  if r["detector_id"] == detector and r["detector_version_id"] == version}
        category_grid.append([
            model.family_short_name, detector_label,
            *[lookup.get(task, "") for task in task_ids],
        ])
    atomic_grid_csv(category_path, category_grid)
    print(f"Updated: {auc_path}")
    print(f"Updated: {category_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--method", help="Detector ID[:VERSION], e.g. cnndet:blur_jpg_prob0_1")
    mode.add_argument("--all", nargs="*", type=Path, metavar="RESULT_DIR",
                      help="Compute method AUCs from score CSVs, then aggregate; defaults to results/{oob,fom,vlm,dib}/")
    mode.add_argument("--aggregate", nargs="*", type=Path, metavar="RESULT_DIR",
                      help="Aggregate existing per-detector AUC CSVs only; defaults to results/{oob,fom,vlm,dib}/")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="Root containing <family>/<detector>/ directories (default: results)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for real-control selection")
    args = parser.parse_args()

    if args.method:
        detector_raw, separator, requested_version = args.method.partition(":")
        try:
            detector = canonical_detector_id(detector_raw)
        except KeyError:
            raise ValueError(f"Unknown detector {args.method!r}; choices: {', '.join(sorted(DET_MODELS))}")
        config = DET_MODELS[detector]
        if config.checkpoint_variants:
            version = requested_version if separator else config.default_variant
            try:
                version, _ = resolve_checkpoints(config, version)
            except ValueError as error:
                raise ValueError(str(error)) from error
        else:
            if separator and requested_version != "default":
                raise ValueError(f"{detector} has no selectable checkpoint versions")
            version = "default"
        result_dir = args.results_root.expanduser().resolve() / config.family / detector
        detector_summary(detector, version, result_dir, args.seed)
    elif args.all is not None:
        paths = args.all or [Path("results") / family for family in FAMILY_ORDER]
        summarize_all(paths, args.seed)
    else:
        paths = args.aggregate or [Path("results") / family for family in FAMILY_ORDER]
        summarize_all(paths, args.seed, recompute=False)


if __name__ == "__main__":
    main()
