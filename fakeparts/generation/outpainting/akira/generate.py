#!/usr/bin/env python3
"""akira adapter using the checked-out official implementation."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_DIR = ADAPTER_DIR / "upstream"
OUTPUT_DIR = ADAPTER_DIR.parents[2] / "FakePartsBench/fake/outpainting/akira"
# This upstream model performs optical camera control, not masked video outpainting.
BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"
MODEL_ID = "xi-wang/akira"
LOCAL_CHECKPOINT = REPO_DIR / "checkpoints/akira.ckpt"
CONFIG_PATH = REPO_DIR / "configs/train_akira/svd_320_576.yaml"
WIDTH = 576
HEIGHT = 320
NUM_FRAMES = 14
FPS = 7
NUM_INFERENCE_STEPS = 25
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 3.0
ORIGINAL_POSE_WIDTH = 1280
ORIGINAL_POSE_HEIGHT = 720
BOKEH_K = 0.0
BOKEH_REPRESENTATION = "sigmoidb"
PLUCKER_TYPE = "cctrl"
DEVICE = "cuda:0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Conditioning image")
    parser.add_argument("--trajectory", type=Path, required=True, help="Camera trajectory in upstream pose format")
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
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from inference_akira import get_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("AKiRa requires CUDA")
    checkpoint = LOCAL_CHECKPOINT
    if not checkpoint.is_file():
        root = Path(snapshot_download(MODEL_ID, allow_patterns=["*.ckpt", "*.pth"]))
        candidates = list(root.rglob("*.ckpt")) + list(root.rglob("*.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"Place the camera-adapter checkpoint at {LOCAL_CHECKPOINT}; "
                               f"Hub snapshot contains {len(candidates)} possible checkpoints")
        checkpoint = candidates[0]
    cfg = OmegaConf.load(CONFIG_PATH)
    return get_pipeline(BASE_MODEL_ID, cfg.unet_subfolder, cfg.down_block_types,
                        cfg.up_block_types, cfg.pose_encoder_kwargs,
                        cfg.attention_processor_kwargs, str(checkpoint), False, DEVICE)


def generate(args, work):
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from diffusers.utils import export_to_video
    from akira.utils.ray_utils import Trajectory

    pose_args = SimpleNamespace(image_width=WIDTH, image_height=HEIGHT,
                                video_length=NUM_FRAMES, ori_width=ORIGINAL_POSE_WIDTH,
                                ori_height=ORIGINAL_POSE_HEIGHT, gpu_id=0,
                                is_rectify_intrinsic=True, get_relative_pose=True,
                                plucker_type=PLUCKER_TYPE)
    trajectory = Trajectory()
    if not trajectory.read_from_file(str(args.trajectory), pose_args, read_lines=NUM_FRAMES):
        raise ValueError(f"Trajectory must contain at least {NUM_FRAMES} valid camera poses")
    trajectory.apply_relative_pose()
    for camera in trajectory.cameras:
        camera.oc_info.bokeh_K = BOKEH_K
        camera.oc_info.bokeh_focus_x = 0.0
        camera.oc_info.bokeh_focus_y = 0.0
    pose = trajectory.get_plucker_embeddings_bokehs(
        device="cpu", bokeh_reprez=BOKEH_REPRESENTATION).contiguous().to(DEVICE)
    with Image.open(args.input) as im:
        image = im.convert("RGB")
    pipeline = load_pipeline()
    with torch.inference_mode():
        frames = pipeline(image=image, pose_embedding=pose, height=HEIGHT, width=WIDTH,
                          num_frames=NUM_FRAMES, num_inference_steps=NUM_INFERENCE_STEPS,
                          min_guidance_scale=MIN_GUIDANCE_SCALE,
                          max_guidance_scale=MAX_GUIDANCE_SCALE, fps=FPS,
                          do_image_process=True,
                          generator=torch.Generator(device=DEVICE).manual_seed(args.seed),
                          output_type="pil").frames[0]
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
