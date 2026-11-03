#!/usr/bin/env python3
"""rave repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR.parents[2] / "FakePartsBench/fake/style/rave"
MODEL_ID = "None"  # Upstream sentinel for the standard SD 1.5 model.
HF_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-depth"
PREPROCESSOR = "depth_midas"
PARAMS = dict(batch_size=4, batch_size_vae=1, cond_step_start=0.0,
              controlnet_conditioning_scale=1.0, controlnet_guidance_end=1.0,
              controlnet_guidance_start=0.0, give_control_inversion=True,
              grid_size=3, sample_size=-1, pad=1, guidance_scale=7.5,
              inversion_prompt="", is_ddim_inversion=True, is_shuffle=True,
              negative_prompts="", num_inference_steps=50, num_inversion_step=50)
DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("--prompt-id must contain only letters, digits, underscores or hyphens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if not args.prompt.strip():
        parser.error("--prompt must not be blank")
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error("--input must point to an existing file")
    args.output_dir = args.output_dir.resolve()
    return args


def load_pipeline():
    import torch
    from pipelines.sd_controlnet_rave import RAVE

    if not torch.cuda.is_available():
        raise RuntimeError("RAVE requires a CUDA GPU")
    pipeline = RAVE(torch.device(DEVICE))
    pipeline.init_models(CONTROLNET_ID, HF_MODEL, PREPROCESSOR, MODEL_ID)
    return pipeline


def generate(args, work):
    import cv2
    import imageio.v2 as imageio
    import math
    import numpy as np
    from utils.video_grid_utils import prepare_video_to_grid

    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot read input video FPS")
    grids = prepare_video_to_grid(str(args.input), PARAMS["sample_size"],
                                  PARAMS["grid_size"], PARAMS["pad"])
    if not grids:
        raise ValueError("RAVE requires at least one full 3x3 grid (9 input frames)")
    for name in ("inverses", "controls"):
        (work / name).mkdir()
    params = dict(PARAMS, seed=args.seed, positive_prompts=args.prompt,
                  image_pil_list=grids, sample_size=len(grids),
                  inverse_path=str(work / "inverses"), control_path=str(work / "controls"))
    frames, _ = load_pipeline()(params)
    if not frames:
        raise RuntimeError("RAVE returned no frames")
    output = work / "result.mp4"
    with imageio.get_writer(str(output), fps=fps / PARAMS["pad"], macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
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
