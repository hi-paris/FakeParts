#!/usr/bin/env python3
"""Compute FakeParts' FVD_W and FVD_FP with the content-debiased-fvd implementation."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import random
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mp4", ".webm", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FakeParts FVD metrics, optionally using TempoFunk WebVid validation for FVD_W."
    )
    parser.add_argument("--fake-dir", type=Path, required=True, help="Generated video directory; searched recursively.")
    webvid_source = parser.add_mutually_exclusive_group()
    webvid_source.add_argument("--webvid-real-dir", type=Path, help="Local WebVid reference-video directory.")
    webvid_source.add_argument(
        "--webvid-dataset",
        default=None,
        help="Stream WebVid URL metadata from an HF dataset repo (e.g. TempoFunk/webvid-10M).",
    )
    parser.add_argument("--webvid-split", default="validation", help="HF WebVid split (default: validation).")
    parser.add_argument("--webvid-download-workers", type=int, default=16, help="Concurrent temporary reference-video downloads.")
    parser.add_argument("--fakeparts-real-dir", type=Path, required=True, help="FakeParts real-reference directory.")
    parser.add_argument("--output", type=Path, help="Path for the JSON result (default: results/fvd/<run-name>.json).")
    parser.add_argument("--work-dir", type=Path, help="Persistent symlink/cache workspace; defaults to /tmp/fakeparts-fvd/<output-stem>.")
    parser.add_argument("--run-name", help="Unique workspace name; defaults to output filename stem.")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Videos per set (default: 10,000).")
    parser.add_argument("--model", choices=("i3d", "videomae"), default="i3d", help="Feature backbone; i3d is the standard FVD setting.")
    parser.add_argument("--checkpoint", type=Path, help="Optional pretrained model checkpoint.")
    parser.add_argument("--resolution", type=int, default=128, help="Decoded clip resolution before model preprocessing (default: 128).")
    parser.add_argument("--sequence-length", type=int, default=16, help="Contiguous frames sampled per video (default: 16).")
    parser.add_argument("--sample-every-n-frames", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Video loader batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu (default: cuda).")
    parser.add_argument("--half-precision", action="store_true", help="Use half precision for feature extraction.")
    return parser.parse_args()


def collect_videos(path: Path) -> list[Path]:
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a video directory: {path}")
    videos = sorted(
        (item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS),
        key=str,
    )
    return videos


def select_videos(videos: list[Path], count: int, seed: int, label: str) -> list[Path]:
    if len(videos) < count:
        raise ValueError(f"{label} has {len(videos)} videos; {count} are required. Refusing to repeat videos.")
    return random.Random(seed).sample(videos, count)


def create_symlink_index(videos: list[Path], link_dir: Path) -> None:
    link_dir.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(videos):
        link = link_dir / f"{index:08d}{source.suffix.lower()}"
        link.symlink_to(source)


def load_webvid_validation_rows(dataset_id: str, split: str) -> list[dict[str, str]]:
    # TempoFunk/webvid-10M stores the validation metadata under data/val. Its
    # CSV contains URLs rather than embedded video bytes, so only this small
    # metadata file is fetched before selecting the requested references.
    if dataset_id.lower() == "tempofunk/webvid-10m" and split.lower() in {"validation", "val"}:
        csv_url = (
            "https://huggingface.co/datasets/TempoFunk/webvid-10M/resolve/main/"
            "data/val/partitions/0000.csv"
        )
    else:
        raise ValueError(
            "Online WebVid loading currently supports TempoFunk/webvid-10M validation (data/val)."
        )
    request = urllib.request.Request(csv_url, headers={"User-Agent": "FakeParts-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "contentUrl" not in rows[0]:
        raise ValueError(f"No contentUrl column found in {dataset_id}/{split} metadata.")
    return rows


def _download_video(row: dict[str, str], destination: Path) -> tuple[Path | None, str | None]:
    url = row.get("contentUrl", "")
    if not url.startswith(("https://", "http://")):
        return None, "missing or invalid contentUrl"
    request = urllib.request.Request(url, headers={"User-Agent": "FakeParts-evaluation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if destination.stat().st_size < 1024:
            return None, "response too small to be a usable video"
        return destination, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def download_webvid_subset(
    dataset_id: str,
    split: str,
    count: int,
    seed: int,
    destination: Path,
    workers: int,
) -> tuple[list[Path], dict[str, int]]:
    rows = load_webvid_validation_rows(dataset_id, split)
    if len(rows) < count:
        raise ValueError(f"{dataset_id}/{split} contains {len(rows)} rows; {count} are required.")
    candidates = random.Random(seed).sample(rows, len(rows))
    destination.mkdir(parents=True, exist_ok=False)
    downloaded: list[Path] = []
    selected_paths: set[Path] = set()
    failed = 0
    attempts = 0
    # Work in bounded waves so failed URLs do not cause unbounded temporary
    # storage use, while keeping network and GPU preparation efficient.
    wave_size = max(workers * 4, 32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(candidates), wave_size):
            if len(downloaded) >= count:
                break
            wave = candidates[start : start + wave_size]
            futures = {
                pool.submit(
                    _download_video,
                    row,
                    destination / f"{start + offset:06d}.mp4",
                ): row
                for offset, row in enumerate(wave)
            }
            successes: list[Path] = []
            for future in concurrent.futures.as_completed(futures):
                path, error = future.result()
                attempts += 1
                if path is None:
                    failed += 1
                else:
                    successes.append(path)
            successes.sort()
            remaining = max(0, count - len(downloaded))
            downloaded.extend(successes[:remaining])
            selected_paths.update(successes[:remaining])
            print(f"WebVid validation references: {len(downloaded)}/{count} ready ({failed} failed URLs)", flush=True)
    if len(downloaded) != count:
        raise RuntimeError(
            f"Only downloaded {len(downloaded)}/{count} videos from {dataset_id}/{split}; "
            f"tried {attempts} URLs ({failed} failed)."
        )
    return downloaded, {
        "available_rows": len(rows),
        "download_attempts": attempts,
        "failed_urls": failed,
        "selected_video_count": len(selected_paths),
    }


def main() -> None:
    args = parse_args()
    if args.n_samples < 2:
        raise ValueError("--n-samples must be at least 2 to estimate feature covariance.")
    if args.resolution < 1 or args.sequence_length < 1 or args.sample_every_n_frames < 1:
        raise ValueError("Resolution and frame sampling arguments must be positive.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative.")
    if args.webvid_download_workers < 1:
        raise ValueError("--webvid-download-workers must be positive.")

    run_name = args.run_name or args.fake_dir.expanduser().name or datetime.now().strftime("fvd_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("--run-name may contain only letters, digits, dot, underscore, and hyphen.")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (Path("results/fvd") / f"{run_name}.json").resolve()
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or Path("/tmp/fakeparts-fvd") / run_name).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)

    sources = {
        "fake": args.fake_dir.expanduser().resolve(),
        "fakeparts_real": args.fakeparts_real_dir.expanduser().resolve(),
    }
    webvid_source_info = None
    webvid_videos: list[Path] | None = None
    if args.webvid_real_dir is not None:
        sources["webvid"] = args.webvid_real_dir.expanduser().resolve()
    elif args.webvid_dataset is not None:
        webvid_dir = work_dir / "webvid_downloads"
        webvid_videos, webvid_source_info = download_webvid_subset(
            args.webvid_dataset,
            args.webvid_split,
            args.n_samples,
            args.seed,
            webvid_dir,
            args.webvid_download_workers,
        )
        sources["webvid"] = webvid_dir
    selected: dict[str, list[Path]] = {}
    source_counts: dict[str, int] = {}
    for label, source_dir in sources.items():
        videos = webvid_videos if label == "webvid" and args.webvid_dataset is not None else collect_videos(source_dir)
        if videos is None:
            raise RuntimeError("WebVid streaming source was not prepared.")
        source_counts[label] = len(videos)
        selected[label] = videos if label == "webvid" and args.webvid_dataset is not None else select_videos(videos, args.n_samples, args.seed, label)

    link_dirs = {label: work_dir / label for label in sources}
    for label, videos in selected.items():
        create_symlink_index(videos, link_dirs[label])

    fvd_root = Path(__file__).resolve().parent / "content-debiased-fvd"
    if not (fvd_root / "cdfvd" / "fvd.py").is_file():
        raise FileNotFoundError(f"content-debiased-fvd submodule is not initialized: {fvd_root}")
    sys.path.insert(0, str(fvd_root))

    import torch
    from cdfvd.fvd import cdfvd
    from cdfvd.utils.metric_utils import FeatureStats

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or install CUDA-enabled PyTorch.")

    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        checkpoint_name = (
            "i3d_pretrained_400.pt"
            if args.model == "i3d"
            else "vit_g_hybrid_pt_1200e_ssv2_ft.pth"
        )
        checkpoint = Path.home() / ".cache" / "fakeparts" / "content-debiased-fvd" / checkpoint_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    evaluator = cdfvd(
        model=args.model,
        n_real=args.n_samples,
        n_fake=args.n_samples,
        ckpt_path=str(checkpoint),
        seed=args.seed,
        compute_feats=False,
        device=args.device,
        half_precision=args.half_precision,
    )
    video_loader_args = {
        "resolution": args.resolution,
        "sequence_length": args.sequence_length,
        "sample_every_n_frames": args.sample_every_n_frames,
        "data_type": "video_folder",
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
    }

    print(f"Extracting generated-video features ({args.n_samples} clips)...")
    fake_loader = evaluator.load_videos(str(link_dirs["fake"]), **video_loader_args)
    fake_stats = evaluator.compute_fake_stats(fake_loader)
    if fake_stats.num_items != args.n_samples:
        raise RuntimeError(f"Expected {args.n_samples} generated features, got {fake_stats.num_items}.")

    scores: dict[str, float] = {}
    references = [("FVD_FP", "fakeparts_real")]
    if "webvid" in sources:
        references.insert(0, ("FVD_W", "webvid"))
    for output_key, reference_key in references:
        print(f"Extracting {reference_key} real-reference features ({args.n_samples} clips)...")
        evaluator.real_stats = FeatureStats(capture_mean_cov=True, max_items=args.n_samples)
        real_loader = evaluator.load_videos(str(link_dirs[reference_key]), **video_loader_args)
        real_stats = evaluator.compute_real_stats(real_loader)
        if real_stats.num_items != args.n_samples:
            raise RuntimeError(f"Expected {args.n_samples} {reference_key} features, got {real_stats.num_items}.")
        scores[output_key] = float(evaluator.compute_fvd_from_stats(fake_stats=fake_stats, real_stats=real_stats))
        print(f"{output_key}: {scores[output_key]:.6f}")

    try:
        fvd_revision = __import__("subprocess").run(
            ["git", "-C", str(fvd_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        fvd_revision = None

    result = {
        **scores,
        "config": {
            "implementation": "songweige/content-debiased-fvd",
            "implementation_revision": fvd_revision,
            "feature_model": args.model,
            "checkpoint": str(checkpoint),
            "fake_samples": args.n_samples,
            "webvid_reference_samples": args.n_samples,
            "fakeparts_real_reference_samples": args.n_samples,
            "available_source_video_counts": source_counts,
            "seed": args.seed,
            "resolution": args.resolution,
            "sequence_length": args.sequence_length,
            "sample_every_n_frames": args.sample_every_n_frames,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "half_precision": args.half_precision,
            "video_selection": "uniform random sample without replacement; fixed seed",
            "frame_selection": "one contiguous clip per video; fixed seed",
            "input_directories": {key: str(path) for key, path in sources.items()},
            "webvid_dataset": args.webvid_dataset,
            "webvid_split": args.webvid_split if args.webvid_dataset else None,
            "webvid_reference_mode": (
                "validation metadata fetched from Hugging Face; selected MP4 URLs downloaded to temporary workspace"
                if args.webvid_dataset
                else "local directory"
                if args.webvid_real_dir
                else None
            ),
            "webvid_download": webvid_source_info,
            "symlink_workspace": str(work_dir),
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_path}")
    print(f"Input symlinks and upstream metadata cache: {work_dir}")


if __name__ == "__main__":
    main()
