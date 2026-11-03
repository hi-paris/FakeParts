#!/usr/bin/env python3
"""framer adapter using the checked-out official implementation."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_DIR = ADAPTER_DIR / "upstream"
OUTPUT_DIR = ADAPTER_DIR.parents[2] / "FakePartsBench/fake/interpolation/framer"
MODEL_ID = "wwen1997/framer_512x320"
BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
WIDTH = 512
HEIGHT = 320
NUM_FRAMES = 14
FPS = 7
NUM_INFERENCE_STEPS = 30
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 3.0
MOTION_BUCKET_ID = 127
NOISE_AUG_STRENGTH = 0.02
DECODE_CHUNK_SIZE = 7
# Endpoint-only interpolation uses the upstream no-trajectory branch.
WITH_CONTROL = False
CONTROLNET_COND_SCALE = 0.0
DTYPE = "float16"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    frames = parser.add_mutually_exclusive_group(required=True)
    frames.add_argument("--input", type=Path, help="Video input; use its first and last frames as endpoints.")
    frames.add_argument("--first-frame", type=Path, help="First endpoint image (requires --last-frame).")
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("Invalid --prompt-id; use letters, digits, underscores or hyphens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.input is None and args.last_frame is None:
        parser.error("Provide --input or both --first-frame and --last-frame")
    if args.input is not None and args.last_frame is not None:
        parser.error("--last-frame is only used with --first-frame")
    for name in ("input", "first_frame", "last_frame"):
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
    from huggingface_hub import snapshot_download
    from models_diffusers.controlnet_svd import ControlNetSVDModel
    from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
    from diffusers.utils.import_utils import is_xformers_available
    if not torch.cuda.is_available():
        raise RuntimeError("Framer requires CUDA")
    if not is_xformers_available():
        raise RuntimeError("Install xformers in the Framer environment")
    model = Path(snapshot_download(MODEL_ID))
    dtype = getattr(torch, DTYPE)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        str(model / "unet"), torch_dtype=dtype, low_cpu_mem_usage=True, custom_resume=True)
    controlnet = ControlNetSVDModel.from_pretrained(str(model / "controlnet")).to(dtype)
    unet.enable_xformers_memory_efficient_attention()
    return StableVideoDiffusionInterpControlPipeline.from_pretrained(
        BASE_MODEL_ID, unet=unet, controlnet=controlnet, low_cpu_mem_usage=False,
        torch_dtype=dtype, variant="fp16").to("cuda")


def generate(args, work):
    import torch
    from PIL import Image
    from diffusers.utils import export_to_video
    with Image.open(args.first_frame) as im:
        first = im.convert("RGB").resize((WIDTH, HEIGHT))
    with Image.open(args.last_frame) as im:
        last = im.convert("RGB").resize((WIDTH, HEIGHT))
    pipe = load_pipeline()
    frames = pipe(first, last, with_control=WITH_CONTROL,
                  controlnet_cond_scale=CONTROLNET_COND_SCALE, with_id_feature=False,
                  height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                  num_inference_steps=NUM_INFERENCE_STEPS, fps=FPS,
                  min_guidance_scale=MIN_GUIDANCE_SCALE, max_guidance_scale=MAX_GUIDANCE_SCALE,
                  motion_bucket_id=MOTION_BUCKET_ID, noise_aug_strength=NOISE_AUG_STRENGTH,
                  decode_chunk_size=DECODE_CHUNK_SIZE,
                  generator=torch.Generator(device="cuda").manual_seed(args.seed)).frames[0]
    output = work / "result.mp4"
    export_to_video(frames, str(output), fps=FPS)
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
            work = Path(temp)
            if args.input is not None:
                import cv2
                from PIL import Image
                cap = cv2.VideoCapture(str(args.input))
                try:
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if count < 2:
                        raise ValueError(f"Framer needs at least two input frames: {args.input}")
                    ok, first = cap.read()
                    if not ok:
                        raise ValueError(f"Could not read first input frame: {args.input}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
                    ok, last = cap.read()
                    if not ok:
                        raise ValueError(f"Could not read last input frame: {args.input}")
                    args.first_frame = work / "first.png"
                    args.last_frame = work / "last.png"
                    Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)).save(args.first_frame)
                    Image.fromarray(cv2.cvtColor(last, cv2.COLOR_BGR2RGB)).save(args.last_frame)
                finally:
                    cap.release()
            result = Path(generate(args, work))
            if not result.is_file() or not result.stat().st_size:
                raise RuntimeError(f"No output video produced: {result}")
            os.link(result, output)
    finally:
        os.chdir(previous)
        sys.path.remove(str(REPO_DIR))
    print(f"Saved generated video to {output}")


if __name__ == "__main__":
    main()
