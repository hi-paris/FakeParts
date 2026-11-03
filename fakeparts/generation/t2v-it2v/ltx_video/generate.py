#!/usr/bin/env python3
"""LTX-Video text-to-video adapter.

Model: https://huggingface.co/Lightricks/LTX-Video
API: https://huggingface.co/docs/diffusers/api/pipelines/ltx_video
"""

import argparse
from pathlib import Path
import re


NUM_VIDEOS_PER_PROMPT = 1
# Fixed benchmark profile. Change generation settings here, not through CLI flags.
MODEL_ID = "Lightricks/LTX-Video-0.9.7-distilled"
UPSCALER_ID = "Lightricks/ltxv-spatial-upscaler-0.9.7"
DTYPE = "bfloat16"
FPS = 25
HEIGHT = 720
WIDTH = 1280
DOWNSCALE_FACTOR = 2 / 3
UPSCALE_FACTOR = 2
NUM_FRAMES = 129
TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
REFINE_TIMESTEPS = [1000, 909, 725, 421, 0]
GUIDANCE_SCALE = 1.0
GUIDANCE_RESCALE = 0.7
DECODE_TIMESTEP = 0.05
DECODE_NOISE_SCALE = 0.025
IMAGE_COND_NOISE_SCALE = 0.0
DENOISE_STRENGTH = 0.999
ADAIN_FACTOR = 1.0
FRAME_RATE = 25
MAX_SEQUENCE_LENGTH = 256
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
OUTPUT_DIR = Path("FakePartsBench/fake/t2v/ltx_video")


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
    from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This adapter requires a CUDA GPU.")
    pipe = LTXConditionPipeline.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, DTYPE))
    upsampler = LTXLatentUpsamplePipeline.from_pretrained(
        UPSCALER_ID, vae=pipe.vae, torch_dtype=getattr(torch, DTYPE)
    )
    # Shared VAE: use the official CUDA placement without competing offload hooks.
    pipe.to("cuda")
    upsampler.to("cuda")
    pipe.vae.enable_tiling()
    return pipe, upsampler


def generate(pipe, prompt: str, seed: int):
    import torch
    from PIL import Image

    pipeline, upsampler = pipe
    base_height = int(HEIGHT * DOWNSCALE_FACTOR)
    base_width = int(WIDTH * DOWNSCALE_FACTOR)
    base_height -= base_height % pipeline.vae_spatial_compression_ratio
    base_width -= base_width % pipeline.vae_spatial_compression_ratio
    params = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_frames=NUM_FRAMES,
                  frame_rate=FRAME_RATE, guidance_scale=GUIDANCE_SCALE,
                  guidance_rescale=GUIDANCE_RESCALE, decode_timestep=DECODE_TIMESTEP,
                  decode_noise_scale=DECODE_NOISE_SCALE,
                  image_cond_noise_scale=IMAGE_COND_NOISE_SCALE,
                  max_sequence_length=MAX_SEQUENCE_LENGTH,
                  num_videos_per_prompt=NUM_VIDEOS_PER_PROMPT)
    latents = pipeline(
        **params, height=base_height, width=base_width, timesteps=TIMESTEPS,
        generator=torch.Generator(device="cpu").manual_seed(seed), output_type="latent"
    ).frames
    latents = upsampler(latents=latents, adain_factor=ADAIN_FACTOR, output_type="latent").frames
    frames = pipeline(
        **params, height=base_height * UPSCALE_FACTOR, width=base_width * UPSCALE_FACTOR,
        timesteps=REFINE_TIMESTEPS, denoise_strength=DENOISE_STRENGTH, latents=latents,
        generator=torch.Generator(device="cpu").manual_seed(seed), output_type="pil"
    ).frames[0]
    return [frame.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS) for frame in frames]


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
