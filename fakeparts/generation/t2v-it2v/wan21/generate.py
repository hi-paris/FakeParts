#!/usr/bin/env python3
"""Wan2.1 T2V and IT2V adapters with separate fixed profiles.

Models:
https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
API: https://huggingface.co/docs/diffusers/api/pipelines/wan
"""

import argparse
from pathlib import Path
import re


NUM_VIDEOS_PER_PROMPT = 1
# Fixed benchmark profiles; IT2V uses a fixed landscape canvas and center crop.
DTYPE = "bfloat16"
VAE_DTYPE = "float32"
IMAGE_ENCODER_DTYPE = "float32"
NEGATIVE_PROMPT = "Blurry footage, compression damage, flicker, illegible text, distorted anatomy."
PROFILES = {
    "t2v": {
        "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 15,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "max_sequence_length": 512,
    },
    "it2v": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "max_sequence_length": 512,
    },
}
OUTPUT_ROOT = Path("FakePartsBench/fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PROFILES, required=True)
    parser.add_argument("--input", type=Path, help="Conditioning image, required for IT2V.")
    parser.add_argument("--prompt-id", required=True, help="ID used in {prompt_id}_{seed}.mp4.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("--prompt-id must contain only letters, digits, underscores or hyphens")
    if not args.prompt.strip():
        parser.error("--prompt must not be blank")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    if args.task == "it2v" and (args.input is None or not args.input.is_file()):
        parser.error("IT2V requires --input pointing to an existing image")
    if args.task == "t2v" and args.input is not None:
        parser.error("--input is only supported for IT2V")
    if args.output_dir is None:
        args.output_dir = OUTPUT_ROOT / args.task / "wan21"
    return args


def load_pipeline(task: str):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This adapter requires a CUDA GPU.")
    model_id = PROFILES[task]["model_id"]
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=getattr(torch, VAE_DTYPE)
    )
    if task == "it2v":
        from transformers import CLIPVisionModel

        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=getattr(torch, IMAGE_ENCODER_DTYPE)
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, image_encoder=image_encoder, torch_dtype=getattr(torch, DTYPE)
        )
    else:
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=getattr(torch, DTYPE))
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def prepare_image(input_path: Path, task: str):
    from diffusers.utils import load_image
    from PIL import Image, ImageOps

    profile = PROFILES[task]
    image = load_image(str(input_path)).convert("RGB")
    return ImageOps.fit(
        image, (profile["width"], profile["height"]), method=Image.Resampling.LANCZOS
    )


def generate(pipe, prompt: str, seed: int, task: str, image=None):
    import torch

    if task == "it2v" and image is None:
        raise ValueError("IT2V requires a conditioning image")
    if task == "t2v" and image is not None:
        raise ValueError("T2V does not accept a conditioning image")
    profile = PROFILES[task]
    params = {key: value for key, value in profile.items() if key not in {"model_id", "fps"}}
    conditioning = {"image": image} if task == "it2v" else {}
    return pipe(
        **params,
        **conditioning,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
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

    image = prepare_image(args.input, args.task) if args.task == "it2v" else None
    pipe = load_pipeline(args.task)
    frames = generate(pipe, args.prompt, args.seed, args.task, image)
    export_to_video(frames, str(output_path), fps=PROFILES[args.task]["fps"])
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
