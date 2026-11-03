#!/usr/bin/env python3
"""propainter adapter using the checked-out official implementation."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_DIR = ADAPTER_DIR / "upstream"
OUTPUT_DIR = ADAPTER_DIR.parents[2] / "FakePartsBench/fake/inpainting/propainter"
PARAMS = dict(resize_ratio=1.0, height=-1, width=-1, mask_dilation=4,
              ref_stride=10, neighbor_length=10, subvideo_length=80, raft_iter=20,
              scale_h=1.2, scale_w=1.2, save_fps=24)
USE_FP16 = True
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--task", choices=["inpainting", "outpainting"], default="inpainting")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("Invalid --prompt-id; use letters, digits, underscores or hyphens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    for name in ("input", "mask", "first_frame", "last_frame", "trajectory"):
        value = getattr(args, name, None)
        if value is not None:
            value = value.resolve()
            if not value.is_file():
                parser.error(f"--{name.replace('_', '-')} must point to a file")
            setattr(args, name, value)
    if hasattr(args, "prompt") and not args.prompt.strip():
        parser.error("--prompt cannot be blank")
    if args.task == "inpainting" and args.mask is None:
        parser.error("--mask is required for inpainting")
    if args.task == "outpainting":
        if args.mask is not None:
            parser.error("Outpainting constructs its own mask; omit --mask")
        if args.output_dir == OUTPUT_DIR:
            args.output_dir = OUTPUT_DIR.parents[1] / "outpainting/propainter"
    args.output_dir = args.output_dir.resolve()
    return args


def prepare_mask(path, video, work):
    import cv2

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return path
    mask_dir = work / "masks"
    mask_dir.mkdir()
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not cv2.imwrite(str(mask_dir / f"{count:06d}.png"), gray):
                raise RuntimeError("Could not write staged mask")
            count += 1
    finally:
        capture.release()
    capture = cv2.VideoCapture(str(video))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count == 0 or count != length:
        raise ValueError(f"Mask has {count} frames; input video has {length}")
    return mask_dir


def generate(args, work):
    import random
    import runpy
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    command = [str(REPO_DIR / "inference_propainter.py"), "--video", str(args.input),
               "--output", str(work), "--mode", "video_" + args.task]
    if args.mask is not None:
        command += ["--mask", str(prepare_mask(args.mask, args.input, work))]
    for key, value in PARAMS.items():
        command += ["--" + key, str(value)]
    if USE_FP16:
        command.append("--fp16")
    previous_argv = sys.argv
    try:
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")
    finally:
        sys.argv = previous_argv
    outputs = list(work.rglob("inpaint_out.mp4"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output, got {len(outputs)}")
    return outputs[0]


def main():
    args = parse_args()
    output = args.output_dir / f"{args.prompt_id}_{args.seed}.mp4"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not REPO_DIR.is_dir():
        raise FileNotFoundError(f"Missing upstream checkout: {REPO_DIR}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    sys.path.insert(0, str(REPO_DIR))
    try:
        os.chdir(REPO_DIR)
        with tempfile.TemporaryDirectory(prefix=".generate-", dir=args.output_dir) as temp:
            result = Path(generate(args, Path(temp)))
            if not result.is_file() or not result.stat().st_size:
                raise RuntimeError(f"No output video produced: {result}")
            os.link(result, output)
    finally:
        os.chdir(previous)
        sys.path.remove(str(REPO_DIR))
    print(f"Saved generated video to {output}")


if __name__ == "__main__":
    main()
