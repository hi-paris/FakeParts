#!/usr/bin/env python3
"""Run the three VBench dimensions reported by FakeParts: Cons., Flick., Qual."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


VBENCH_DIMENSIONS = {
    "Cons.": "subject_consistency",
    "Flick.": "temporal_flickering",
    "Qual.": "imaging_quality",
}
VIDEO_EXTENSIONS = {".mp4", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VBench and summarize Cons., Flick., and Qual. for a video file or folder."
    )
    parser.add_argument("--videos", type=Path, required=True, help="Video file or folder (searched recursively).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/vbench"), help="Directory for VBench raw results and summary JSON (default: results/vbench).")
    parser.add_argument("--work-dir", type=Path, help="Directory for symlinked inputs; defaults to /tmp/fakeparts-vbench/<run-name>.")
    parser.add_argument("--cache-dir", type=Path, help="VBench model cache; defaults to ~/.cache/vbench.")
    parser.add_argument("--run-name", help="Unique output name; defaults to a timestamp.")
    parser.add_argument("--gpu", type=int, default=0, help="Single CUDA device index (default: 0).")
    parser.add_argument("--master-port", type=int, default=29500, help="Distributed initialization port; use a unique port for concurrent processes.")
    parser.add_argument("--dimensions", nargs="+", choices=tuple(VBENCH_DIMENSIONS.values()), default=list(VBENCH_DIMENSIONS.values()), help="VBench dimensions to compute (default: all three).")
    parser.add_argument("--imaging-quality-preprocessing", choices=("longer", "shorter", "shorter_centercrop", "None"), default="longer")
    parser.add_argument("--local-checkpoints", action="store_true", help="Use VBench's local-checkpoint loading mode.")
    return parser.parse_args()


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = [item for item in path.rglob("*") if item.is_file()]
    else:
        raise FileNotFoundError(f"Input does not exist: {path}")
    videos = sorted((item.resolve() for item in candidates if item.suffix.lower() in VIDEO_EXTENSIONS), key=str)
    if not videos:
        raise ValueError(f"No supported videos (.mp4/.gif) found under: {path}")
    return videos


def create_input_links(videos: list[Path], link_dir: Path) -> dict[str, str]:
    link_dir.mkdir(parents=True, exist_ok=False)
    path_map: dict[str, str] = {}
    for index, source in enumerate(videos):
        link = link_dir / f"{index:08d}{source.suffix.lower()}"
        link.symlink_to(source)
        path_map[str(link.resolve())] = str(source)
    return path_map


def rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_paths(item, path_map) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, path_map) for item in value]
    if isinstance(value, str):
        try:
            resolved = str(Path(value).resolve())
        except (OSError, RuntimeError):
            return value
        return path_map.get(resolved, value)
    return value


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("vbench_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("--run-name may contain only letters, digits, dot, underscore, and hyphen.")

    videos = collect_videos(args.videos.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("_eval_results.json", "_full_info.json", "_summary.json"):
        candidate = output_dir / f"{run_name}{suffix}"
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation output: {candidate}")
    work_dir = (args.work_dir or Path("/tmp/fakeparts-vbench") / run_name).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    link_dir = work_dir / "videos"
    path_map = create_input_links(videos, link_dir)

    cache_dir = (args.cache_dir or Path.home() / ".cache" / "vbench").expanduser().resolve()
    os.environ["VBENCH_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", str(args.gpu))
    os.environ.setdefault("WORLD_SIZE", "1")

    vbench_root = Path(__file__).resolve().parent / "VBench"
    if not (vbench_root / "vbench" / "__init__.py").is_file():
        raise FileNotFoundError(f"VBench submodule is not initialized: {vbench_root}")
    sys.path.insert(0, str(vbench_root))

    import torch
    from vbench import VBench
    from vbench.distributed import dist_init

    if not torch.cuda.is_available():
        raise RuntimeError("VBench requires CUDA; no CUDA device is available.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This wrapper runs one process on one GPU. Do not invoke it with torchrun.")

    dist_init()
    try:
        evaluator = VBench(
            torch.device("cuda", args.gpu),
            str(vbench_root / "vbench" / "VBench_full_info.json"),
            str(output_dir),
        )
        evaluator.evaluate(
            videos_path=str(link_dir),
            name=run_name,
            prompt_list=[],
            dimension_list=list(args.dimensions),
            local=args.local_checkpoints,
            read_frame=False,
            mode="custom_input",
            imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    raw_results_path = output_dir / f"{run_name}_eval_results.json"
    full_info_path = output_dir / f"{run_name}_full_info.json"
    if not raw_results_path.is_file():
        raise FileNotFoundError(f"VBench did not produce its expected result file: {raw_results_path}")

    raw_results = json.loads(raw_results_path.read_text(encoding="utf-8"))
    raw_results = rewrite_paths(raw_results, path_map)
    raw_results_path.write_text(json.dumps(raw_results, indent=2) + "\n", encoding="utf-8")
    if full_info_path.is_file():
        full_info = json.loads(full_info_path.read_text(encoding="utf-8"))
        full_info_path.write_text(
            json.dumps(rewrite_paths(full_info, path_map), indent=2) + "\n", encoding="utf-8"
        )

    dimension_scores: dict[str, float] = {}
    for short_name, dimension in VBENCH_DIMENSIONS.items():
        if dimension in raw_results:
            result = raw_results[dimension]
            dimension_scores[short_name] = float(result[0])

    summary = {
        "input": str(args.videos.expanduser().resolve()),
        "video_count": len(videos),
        "metrics": dimension_scores,
        "vbench_dimensions": VBENCH_DIMENSIONS,
        "config": {
            "mode": "custom_input",
            "imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing,
            "device": f"cuda:{args.gpu}",
            "dimensions": list(args.dimensions),
            "local_checkpoints": args.local_checkpoints,
        },
        "raw_results": str(raw_results_path),
        "full_info": str(full_info_path),
        "input_symlinks": str(link_dir),
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for metric, score in dimension_scores.items():
        print(f"{metric}: {score:.6f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
