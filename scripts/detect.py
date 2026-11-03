"""Run a detector on FakePartsBench/{fake,real}."""
from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import os
import re
import shlex
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_ERROR_TRACEBACK_PRINTED = False


def _print_debug_traceback() -> None:
    global _ERROR_TRACEBACK_PRINTED
    if os.getenv("FAKEPARTS_DEBUG_TRACEBACK") and not _ERROR_TRACEBACK_PRINTED:
        traceback.print_exc()
        _ERROR_TRACEBACK_PRINTED = True

from fakeparts.detection.config import (Checkpoint, CSV_FIELDS, DEFAULT_DEVICE,
    DEFAULT_THRESHOLD, DEFAULT_WORKERS, DET_MODELS,
    canonical_detector_id, resolve_checkpoints)
from fakeparts.detection.base import is_cuda_oom
from fakeparts.detection.utils import (VideoDataset, collate_videos, ensure_checkpoints,
                                       cache_video_frames, iter_frame_batches,
                                       load_detector, load_inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="[FakeParts] run a checked-in video detector")
    parser.add_argument("dataset", type=Path, nargs="?", help="input file, folder, or FakePartsBench root")
    parser.add_argument("-i", "--input", type=Path,
                        help="one video, folder, benchmark root, or JSONL sample list (or FAKEPARTS_INPUT)")
    parser.add_argument("--input-root", type=Path,
                        help="dataset root for relative paths in a JSONL input")
    parser.add_argument("-f", "--det-family", choices=("oob", "fom", "vlm", "dib"),
                        default=os.getenv("FAKEPARTS_DET_FAMILY"),
                        help="detector family; checked against the selected detector method")
    parser.add_argument("-m", "--det-method", "--model", dest="det_method",
                        default=os.getenv("FAKEPARTS_DET_METHOD", os.getenv("FAKEPARTS_MODEL")),
                        help="registered detector method ID (--model remains a compatibility alias)")
    parser.add_argument("--det-version", "--version", dest="det_version",
                        default=os.getenv("FAKEPARTS_DET_VERSION", os.getenv("FAKEPARTS_VERSION")),
                        help="checkpoint version ID for the selected detector method")
    parser.add_argument(
        "-b", "--batch-size", type=int,
        help="override the detector's VRAM-based batch size (OOM retries reduce it automatically)",
    )
    parser.add_argument("-j", "--workers", type=int, default=DEFAULT_WORKERS,
                        help="reserved for detector data loaders")
    parser.add_argument("--output-dir", type=Path, default=os.getenv("FAKEPARTS_OUTPUT_DIR"))
    parser.add_argument("--ckpt", type=Path, default=os.getenv("FAKEPARTS_CKPT"),
                        help="override the selected version's primary checkpoint")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--frame-cache-dir", type=Path,
                        default=Path(os.environ["FAKEPARTS_FRAME_CACHE_DIR"])
                        if os.environ.get("FAKEPARTS_FRAME_CACHE_DIR") else None,
                        help="shared frame cache (default: /tmp/fakeparts_frames_<uid>/<video-hash>/)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--reference-dir", type=Path,
                        help="real reference video directory required for NSG-VD dMMD variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="run only the first N discovered videos")
    parser.add_argument("--sample-task", action="append", default=[],
                        help="keep JSONL rows for this task; may be repeated")
    parser.add_argument("--sample-method", action="append", default=[],
                        help="keep JSONL rows for this generator method; may be repeated")
    parser.add_argument("--sample-id", action="append", default=[],
                        help="keep this JSONL sample ID; may be repeated")
    parser.add_argument("--sample-label", action="append", default=[],
                        help="keep this JSONL label (real or fake); may be repeated")
    parser.add_argument("--list-models", action="store_true", help="list detector adapters and known versions")
    args = parser.parse_args()
    if ((args.batch_size is not None and args.batch_size < 1) or args.workers < 0
            or (args.limit is not None and args.limit < 1)):
        parser.error("batch sizes must be positive, workers nonnegative, and limit positive")
    return args


def _predict_frame_batch(detector, images: list, batch_size: int) -> list[dict]:
    try:
        results = detector.predict_frames(images)
        if len(results) != len(images):
            raise ValueError("detector returned the wrong number of results")
        return results
    except Exception as error:
        if not is_cuda_oom(error) or len(images) <= 1:
            raise
        smaller = max(1, len(images) // 2)
        previous = getattr(detector, "_fakeparts_frame_batch_size", batch_size)
        detector._fakeparts_frame_batch_size = min(previous, smaller)
        import torch

        torch.cuda.empty_cache()
        print(
            f"[detect] CUDA OOM at frame batch {len(images)}; retrying with "
            f"batch {smaller}", flush=True,
        )
        return (_predict_frame_batch(detector, images[:smaller], smaller)
                + _predict_frame_batch(detector, images[smaller:], smaller))


def write_frame_rows(writer, sample, path: Path, detector, batch_size: int,
                     frame_cache_dir: Path) -> None:
    base = sample.row()
    try:
        def current_batch_size() -> int:
            return getattr(detector, "_fakeparts_frame_batch_size", batch_size)

        for indices, images in iter_frame_batches(path, current_batch_size, frame_cache_dir):
            results = _predict_frame_batch(detector, images, current_batch_size())
            for index, result in zip(indices, results, strict=True):
                writer.writerow({**base, "sample_id": f"{sample.sample_id}_{index}",
                                 "frame": index, **result, "error": ""})
    except Exception as error:
        _print_debug_traceback()
        writer.writerow({**base, "frame": "", "error": str(error)})


def write_video_row(writer, sample, path: Path, detector, work_dir: Path) -> None:
    try:
        result = detector.predict_video_with_workdir(path, work_dir)
        writer.writerow({**sample.row(), **result, "error": ""})
    except Exception as error:
        _print_debug_traceback()
        writer.writerow({**sample.row(), "error": str(error)})


def _cuda_vram_info(device: str) -> tuple[str, float, float] | None:
    """Return the visible GPU name and free/total VRAM in GiB."""
    try:
        import torch

        device_obj = torch.device(device)
        if device_obj.type != "cuda" or not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
        return (
            torch.cuda.get_device_name(device_obj),
            free_bytes / (1024 ** 3),
            total_bytes / (1024 ** 3),
        )
    except (ImportError, RuntimeError, ValueError):
        return None


def _complete_rows_for_output(output: Path, samples, output_level: str,
                              dataset: Path, frame_cache_dir: Path):
    """Load complete sample rows from a finished CSV or earlier run temp files."""
    expected = {sample.rel_path: sample for sample in samples}
    completed = {}
    sources = ([output] if output.is_file() else [])
    sources.extend(sorted(
        output.parent.glob(f"{output.name}.*.tmp"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    ))
    for source in sources:
        rows_by_path = {rel_path: [] for rel_path in expected}
        try:
            with source.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "rel_path" not in reader.fieldnames:
                    continue
                for row in reader:
                    rel_path = row.get("rel_path")
                    if rel_path in rows_by_path and row.get("sample_id"):
                        rows_by_path[rel_path].append(row)
        except (OSError, UnicodeError, csv.Error) as error:
            print(f"Ignoring unreadable partial CSV {source}: {error}", file=sys.stderr)
        for rel_path, sample in expected.items():
            if rel_path in completed:
                continue
            rows = rows_by_path[rel_path]
            successful = [row for row in rows if not row.get("error")]
            if output_level == "video":
                if successful:
                    completed[rel_path] = [successful[0]]
                continue
            if not successful or any(row.get("error") for row in rows):
                continue
            frame_count = sample.frame_count
            if frame_count is None:
                source_path = Path(sample.rel_path)
                if not source_path.is_absolute():
                    source_path = dataset / source_path
                cached = cache_video_frames(source_path, frame_cache_dir)
                frame_count = len(list(cached.glob("frame_*.jpg")))
            frame_rows = {}
            for row in successful:
                try:
                    frame_index = int(row["frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= frame_index < frame_count:
                    frame_rows.setdefault(frame_index, row)
            if len(frame_rows) == frame_count:
                completed[rel_path] = [frame_rows[index] for index in range(frame_count)]
    return completed


def _new_temporary_path(output: Path, run_id: str) -> Path:
    base = output.with_suffix(output.suffix + f".{run_id}.tmp")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output.with_suffix(output.suffix + f".{run_id}.{suffix}.tmp")
        suffix += 1
    return candidate


def _open_temporary_csv(path: Path):
    for attempt in range(5):
        try:
            return path.open("w", encoding="utf-8", newline="")
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == 4:
                raise
            delay = min(2 ** attempt, 8)
            print(
                f"[detect] stale filesystem handle opening {path}; "
                f"retrying in {delay}s ({attempt + 1}/4)",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"could not open temporary CSV: {path}")


def _write_rows(path: Path, fields, rows) -> None:
    run_id = os.getenv("SLURM_JOB_ID") or os.getenv("FAKEPARTS_RUN_ID") or "manual"
    temporary = _new_temporary_path(path, run_id)
    with _open_temporary_csv(temporary) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temporary.replace(path)


def _filename_token(value: str) -> str:
    """Make a metadata identifier safe and stable for CSV filenames."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    if not token:
        raise ValueError(f"cannot make a filename token from {value!r}")
    return token


def _output_groups(samples):
    """Group arbitrary JSONL/folder inputs by dataset identity, never input name."""
    groups = {}
    identities = {}
    for sample in samples:
        label = sample.label.strip().lower()
        if label == "fake":
            task = sample.class_name
            method = sample.method
            path_parts = Path(sample.rel_path).parts
            fake_index = next((i for i, part in enumerate(path_parts) if part.lower() == "fake"), None)
            if fake_index is not None:
                task = task or (path_parts[fake_index + 1] if len(path_parts) > fake_index + 1 else "")
                method = method or (path_parts[fake_index + 2] if len(path_parts) > fake_index + 2 else "")
            if not task or not method:
                raise ValueError(
                    f"cannot name fake result for {sample.rel_path}: task and method metadata are required"
                )
            identity = ("fake", task, method)
            stem = f"{_filename_token(task)}_{_filename_token(method)}"
        elif label == "real":
            source = (sample.method if sample.class_name.strip().lower() == "real"
                      else sample.class_name)
            path_parts = Path(sample.rel_path).parts
            real_index = next((i for i, part in enumerate(path_parts) if part.lower() == "real"), None)
            if real_index is not None and len(path_parts) > real_index + 1:
                source = path_parts[real_index + 1]
            if not source or source.lower() == "real":
                raise ValueError(
                    f"cannot name real result for {sample.rel_path}: real source class metadata is required"
                )
            identity = ("real", source)
            stem = f"real_{_filename_token(source)}"
        else:
            raise ValueError(
                f"cannot name result for {sample.rel_path}: label must be fake or real, got {sample.label!r}"
            )
        prior = identities.setdefault(stem, identity)
        if prior != identity:
            raise ValueError(f"different dataset groups collapse to the same CSV stem {stem!r}")
        groups.setdefault(stem, []).append(sample)
    return groups


def main() -> None:
    args = parse_args()
    if args.list_models:
        for name, config in sorted(DET_MODELS.items()):
            versions = ",".join(
                f"{config.version_name(version)} [{version}]"
                for version in sorted(config.checkpoint_variants)
            ) or "Default [default]"
            print(f"{config.family_full_name}\t{config.family_short_name}\t{config.family}\t{config.full_name}\t{config.short_name}\t{name}\t{versions}\timplemented={str(config.implemented).lower()}\t{config.output_level}")
        return
    if args.dataset is not None and args.input is not None:
        raise SystemExit("provide either positional dataset or --input, not both")
    env_input = os.getenv("FAKEPARTS_INPUT")
    input_path = args.input or args.dataset or (Path(env_input) if env_input else None)
    if input_path is None:
        raise SystemExit("provide an input file/folder or set FAKEPARTS_INPUT")
    if args.det_method is None:
        raise SystemExit("provide --det-method or set FAKEPARTS_DET_METHOD")
    try:
        detector_id = canonical_detector_id(args.det_method)
        config = DET_MODELS[detector_id]
    except KeyError as error:
        raise SystemExit(f"unknown detector method {args.det_method!r}; choose from: {', '.join(DET_MODELS)}") from error
    if args.det_family and args.det_family != config.family:
        raise SystemExit(
            f"detector method {detector_id!r} belongs to family {config.family!r}, "
            f"not {args.det_family!r}"
        )
    try:
        dataset, samples = load_inputs(input_path, args.input_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.sample_task:
        selected_tasks = {value.lower().replace("-", "_") for value in args.sample_task}
        samples = [sample for sample in samples
                   if sample.class_name.lower().replace("-", "_") in selected_tasks]
    if args.sample_method:
        selected_methods = {value.lower().replace("-", "_") for value in args.sample_method}
        samples = [sample for sample in samples
                   if sample.method.lower().replace("-", "_") in selected_methods]
    if args.sample_id:
        selected_ids = set(args.sample_id)
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    if args.sample_label:
        selected_labels = {value.lower() for value in args.sample_label}
        samples = [sample for sample in samples if sample.label.lower() in selected_labels]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit("no videos selected after applying the requested filters")
    try:
        version, checkpoints = resolve_checkpoints(config, args.det_version)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.ckpt:
        checkpoints = (Checkpoint(args.ckpt.expanduser().resolve()), *checkpoints[1:])

    batch_size = args.batch_size
    run_id = os.getenv("SLURM_JOB_ID") or os.getenv("FAKEPARTS_RUN_ID") or "manual"
    frame_cache_dir = (
        args.frame_cache_dir
        or Path("/tmp") / f"fakeparts_frames_{os.getuid()}"
    ).expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  PROJECT_ROOT / "results" / config.family / detector_id)
    try:
        groups = _output_groups(samples)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    version_suffix = _filename_token(version)
    print(
        f"Method: {config.class_name} ({detector_id}); version: {version}; "
        f"videos: {len(samples)}; groups: {len(groups)}; level: {config.output_level}"
    )
    for checkpoint in checkpoints:
        print(f"Checkpoint: {checkpoint.path}")
    if not config.implemented:
        raise SystemExit(f"{config.class_name} adapter is not implemented")

    fields = CSV_FIELDS[config.output_level]
    plans = []
    for stem, group_samples in sorted(groups.items()):
        output = output_dir / f"{stem}.{version_suffix}.csv"
        part_samples = group_samples
        completed_rows = _complete_rows_for_output(
            output, part_samples, config.output_level, dataset, frame_cache_dir)
        completed_rel_paths = set(completed_rows)
        missing_samples = [sample for sample in part_samples
                           if sample.rel_path not in completed_rel_paths]
        retained_rows = [row for sample in part_samples
                         for row in completed_rows.get(sample.rel_path, [])]
        plans.append((output, part_samples, completed_rel_paths,
                      retained_rows, missing_samples))
        print(
            f"CSV: {output}; complete {len(completed_rel_paths)}/{len(part_samples)}; "
            f"remaining {len(missing_samples)}",
            flush=True,
        )
    if args.dry_run:
        for output, _, completed_rel_paths, _, missing_samples in plans:
            action = "skip" if not missing_samples and output.is_file() else "run"
            print(f"Would {action}: {output} ({len(completed_rel_paths)} complete, "
                  f"{len(missing_samples)} remaining)")
        return

    if not any(plan[4] for plan in plans):
        for output, part_samples, _, retained_rows, _ in plans:
            if output.is_file():
                print(f"Skipped complete CSV: {output}")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                _write_rows(output, fields, retained_rows)
                print(f"Restored complete CSV from prior run rows: {output}")
        return

    try:
        ckpt_paths = ensure_checkpoints(checkpoints)
    except (FileNotFoundError, RuntimeError) as error:
        target = detector_id
        if config.checkpoint_variants:
            target = f"{target}:{version}"
        quoted_target = shlex.quote(target)
        setup_script = shlex.quote(str(PROJECT_ROOT / "scripts" / "set_up_detector.sh"))
        print(f"{error}", file=sys.stderr)
        print(
            f"[detect] Prepare the missing weights with: bash {setup_script} --methods {quoted_target}",
            file=sys.stderr,
        )
        print(
            f"[detect] Or submit it with: sbatch {setup_script} --methods {quoted_target}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    if not ckpt_paths:
        raise SystemExit("detector has no checkpoint configured")

    missing_all = [sample for _, _, _, _, missing in plans for sample in missing]
    print(f"Frame cache: {frame_cache_dir}")
    for index, sample in enumerate(missing_all, 1):
        source = Path(sample.rel_path)
        if not source.is_absolute():
            source = dataset / source
        cache_video_frames(source, frame_cache_dir)
        if index % 10 == 0 or index == len(missing_all):
            print(f"Frame cache ready: {index}/{len(missing_all)} videos", flush=True)

    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = frame_cache_dir / "_work" / detector_id / version_suffix
    work_dir.mkdir(parents=True, exist_ok=True)
    with nullcontext(work_dir):
        adapter_kwargs = {}
        if detector_id == "nsg_vd" and args.reference_dir:
            adapter_kwargs["reference_dir"] = args.reference_dir.expanduser().resolve()
        detector = load_detector(
            config, ckpt_paths, device=args.device, threshold=args.threshold,
            crop_size=args.crop_size, work_dir=work_dir, version=version, **adapter_kwargs)
        detector.frame_cache_root = frame_cache_dir
        vram_info = _cuda_vram_info(args.device)
        gpu_name = vram_info[0] if vram_info else None
        free_vram_gb = vram_info[1] if vram_info else None
        if batch_size is None:
            batch_size = config.batch_size_for_free_vram(free_vram_gb)
            selection = "config/VRAM"
        else:
            selection = "CLI override"
        batch_setter = getattr(detector, "set_inference_batch_size", None)
        if config.output_level == "video" and batch_setter is None:
            if args.batch_size is not None:
                print(
                    f"[detect] {args.det_method} processes videos serially; "
                    "--batch-size does not change this adapter's video path.",
                    flush=True,
                )
            batch_size = 1
            selection = "adapter's serial video path"
        detector._fakeparts_frame_batch_size = batch_size
        if batch_setter is not None:
            batch_setter(batch_size)
        if vram_info:
            vram_text = f"{gpu_name}; {vram_info[1]:.1f}/{vram_info[2]:.1f} GiB free"
        else:
            vram_text = "GPU/VRAM unavailable"
        print(
            f"Inference batch: {batch_size} {config.batch_unit} ({selection}; {vram_text})",
            flush=True,
        )
        for output, part_samples, completed_rel_paths, retained_rows, missing_samples in plans:
            if not missing_samples:
                if not output.is_file():
                    _write_rows(output, fields, retained_rows)
                    print(f"Restored complete CSV from prior run rows: {output}")
                else:
                    print(f"Skipped complete CSV: {output}")
                continue
            # Video-level preparation is serialized per video; decoded frames
            # are shared by detector jobs on this node through the stable cache.
            video_batch_size = 1
            loader = DataLoader(VideoDataset(dataset, missing_samples), batch_size=video_batch_size,
                                shuffle=False, num_workers=args.workers,
                                collate_fn=collate_videos)
            run_suffix = os.getenv("SLURM_JOB_ID") or os.getenv("FAKEPARTS_RUN_ID") or "manual"
            temporary = _new_temporary_path(output, run_suffix)
            with _open_temporary_csv(temporary) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(retained_rows)
                completed = len(completed_rel_paths)
                for items in loader:
                    if config.output_level == "frame":
                        for sample, path in items:
                            write_frame_rows(writer, sample, path, detector, batch_size,
                                             frame_cache_dir)
                            completed += 1
                            stream.flush()
                            print(f"Scored video: {completed}/{len(part_samples)}", flush=True)
                    else:
                        for sample, path in items:
                            sample_cache = hashlib.sha256(sample.sample_id.encode()).hexdigest()
                            sample_work_dir = work_dir / sample_cache
                            write_video_row(writer, sample, path, detector, sample_work_dir)
                            completed += 1
                            stream.flush()
                            print(f"Scored video: {completed}/{len(part_samples)}", flush=True)
            temporary.replace(output)
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()
