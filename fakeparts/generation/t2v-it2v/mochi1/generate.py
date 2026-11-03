#!/usr/bin/env python3
"""Mochi-1 Preview text-to-video adapter.

Model: https://huggingface.co/genmo/mochi-1-preview
API: https://huggingface.co/docs/diffusers/api/pipelines/mochi
"""

import argparse
from pathlib import Path
import re


NUM_VIDEOS_PER_PROMPT = 1
# Fixed benchmark profile. Change generation settings here, not through CLI flags.
MODEL_ID = "genmo/mochi-1-preview"
DTYPE = "float32"
AUTOCAST_DTYPE = "bfloat16"
FPS = 25
HEIGHT = 480
WIDTH = 848
NUM_FRAMES = 129
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.5
MAX_SEQUENCE_LENGTH = 256
NEGATIVE_PROMPT = ""
OUTPUT_DIR = Path("FakePartsBench/fake/t2v/mochi1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", required=True, help="ID used in {prompt_id}_{seed}.mp4.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("--prompt-id must contain only letters, digits, underscores or hyphens")
    if not args.prompt.strip():
        parser.error("--prompt must not be blank")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    return args


def load_pipeline():
    import torch
    from diffusers import MochiPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This adapter requires a CUDA GPU.")
    pipe = MochiPipeline.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, DTYPE))
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def generate(pipe, prompt: str, seed: int):
    import torch

    # Full-precision weights with BF16 autocast follow the quality-oriented example.
    with torch.autocast("cuda", getattr(torch, AUTOCAST_DTYPE), cache_enabled=False):
        return pipe(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            negative_prompt=NEGATIVE_PROMPT,
            prompt=prompt,
            num_videos_per_prompt=NUM_VIDEOS_PER_PROMPT,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames[0]


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / f"{args.prompt_id}_{args.seed}.mp4"
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video

    pipe = load_pipeline()
    frames = generate(pipe, args.prompt, args.seed)
    export_to_video(frames, str(output_path), fps=FPS)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
