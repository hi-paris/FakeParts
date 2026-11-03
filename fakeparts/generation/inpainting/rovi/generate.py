#!/usr/bin/env python3
"""rovi adapter using the checked-out official implementation."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_DIR = ADAPTER_DIR / "upstream"
OUTPUT_DIR = ADAPTER_DIR.parents[2] / "FakePartsBench/fake/inpainting/rovi"
MODEL_ID = "jianzongwu/lgvi"
WIDTH = 512
HEIGHT = 320
NUM_FRAMES = 24
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.0
IMAGE_GUIDANCE_SCALE = 1.5
DTYPE = "float16"
LATENT_SCALE = 0.18215


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="Referring edit, e.g. remove the red car")
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

    args.output_dir = args.output_dir.resolve()
    return args


def load_pipeline():
    import torch
    from rovi.models.unet import RoviModel
    from rovi.pipelines.pipeline_rovi import RoviPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("ROVI requires CUDA")
    dtype = getattr(torch, DTYPE)
    unet = RoviModel.from_pretrained(MODEL_ID, subfolder="unet").to("cuda", dtype)
    return RoviPipeline.from_pretrained(MODEL_ID, unet=unet, torch_dtype=dtype).to("cuda")


def generate(args, work):
    import cv2
    import math
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    capture = cv2.VideoCapture(str(args.input))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS")
        for _ in range(NUM_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"ROVI profile requires {NUM_FRAMES} input frames")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(np.asarray(image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)))
    finally:
        capture.release()
    pipe = load_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        pixels = pixels.to(device="cuda", dtype=getattr(torch, DTYPE)) / 127.5 - 1.0
        latents = pipe.vae.encode(pixels).latent_dist.sample(generator=generator)
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0) * LATENT_SCALE
        result = pipe(args.prompt, img_condition=latents, num_inference_steps=NUM_INFERENCE_STEPS,
                      guidance_scale=GUIDANCE_SCALE, image_guidance_scale=IMAGE_GUIDANCE_SCALE,
                      generator=generator).videos
        output_frames = (result[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    output = work / "result.mp4"
    imageio.mimwrite(str(output), output_frames, fps=fps, macro_block_size=1)
    return output


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
