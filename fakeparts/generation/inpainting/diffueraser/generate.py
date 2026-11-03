#!/usr/bin/env python3
"""diffueraser repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR.parents[2] / "FakePartsBench/fake/inpainting/diffueraser"
VIDEO_LENGTH = 10
MAX_IMG_SIZE = 960
MASK_DILATION = 8
REF_STRIDE = 10
NEIGHBOR_LENGTH = 10
SUBVIDEO_LENGTH = 50
NFRAMES = 22
CHECKPOINT = "2-Step"
GUIDANCE_SCALE = 0.0
BLENDED = True
WEIGHTS = REPO_DIR / "weights"
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
PCM_MODEL_ID = "wangfuyun/PCM_Weights"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("--prompt-id must contain only letters, digits, underscores or hyphens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error("--input must point to an existing file")
    args.mask = args.mask.resolve()
    if not args.mask.is_file():
        parser.error("--mask must point to an existing file")
    args.output_dir = args.output_dir.resolve()
    return args


def load_pipeline(work):
    from diffueraser.diffueraser import DiffuEraser
    from propainter.inference import Propainter, get_device
    from huggingface_hub import snapshot_download

    for name in ("diffuEraser", "propainter"):
        if not (WEIGHTS / name).is_dir():
            raise FileNotFoundError(f"Missing weights: {WEIGHTS / name}; see weights/README.md")
    base = str(WEIGHTS / "stable-diffusion-v1-5") if (WEIGHTS / "stable-diffusion-v1-5").is_dir() else BASE_MODEL_ID
    vae = str(WEIGHTS / "sd-vae-ft-mse") if (WEIGHTS / "sd-vae-ft-mse").is_dir() else VAE_MODEL_ID
    pcm = WEIGHTS / "PCM_Weights"
    if not pcm.is_dir():
        pcm = Path(snapshot_download(PCM_MODEL_ID, allow_patterns=["sd15/*"]))
    # Upstream uses a relative PCM path. Stage it without changing upstream code.
    (work / "weights").mkdir()
    (work / "weights/PCM_Weights").symlink_to(pcm, target_is_directory=True)
    device = get_device()
    previous_cwd = Path.cwd()
    try:
        os.chdir(work)
        eraser = DiffuEraser(device, base, vae, str(WEIGHTS / "diffuEraser"), ckpt=CHECKPOINT)
    finally:
        os.chdir(previous_cwd)
    prior = Propainter(str(WEIGHTS / "propainter"), device=device)
    return prior, eraser


def generate(args, work):
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prior, eraser = load_pipeline(work)
    prior_path, output = work / "priori.mp4", work / "result.mp4"
    prior.forward(str(args.input), str(args.mask), str(prior_path),
                  video_length=VIDEO_LENGTH, ref_stride=REF_STRIDE,
                  neighbor_length=NEIGHBOR_LENGTH, subvideo_length=SUBVIDEO_LENGTH,
                  mask_dilation=MASK_DILATION)
    eraser.forward(str(args.input), str(args.mask), str(prior_path), str(output),
                   max_img_size=MAX_IMG_SIZE, video_length=VIDEO_LENGTH,
                   mask_dilation_iter=MASK_DILATION, nframes=NFRAMES,
                   seed=args.seed, guidance_scale=GUIDANCE_SCALE, blended=BLENDED)
    return output


def main():
    args = parse_args()
    output = args.output_dir / f"{args.prompt_id}_{args.seed}.mp4"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(REPO_DIR))
    try:
        os.chdir(REPO_DIR)
        with tempfile.TemporaryDirectory(prefix=".generate-", dir=args.output_dir) as temp:
            result = Path(generate(args, Path(temp)))
            if not result.is_file() or result.stat().st_size == 0:
                raise RuntimeError(f"Repository did not produce a video: {result}")
            # Exclusive creation prevents overwriting another run's output.
            with result.open("rb") as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(str(REPO_DIR))
    print(f"Saved generated video to {output}")


if __name__ == "__main__":
    main()
