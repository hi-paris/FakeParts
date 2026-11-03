#!/usr/bin/env python3
"""hunyuanportrait repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR.parents[3] / "FakePartsBench/fake/faceswap/hunyuanportrait"
# Architecture and checkpoint paths come from the repository's official config.
CONFIG_PATH = REPO_DIR / "config/hunyuan-portrait.yaml"
PARAMS = dict(width=512, height=512, num_inference_steps=25, pad_frames=15,
              n_sample_frames=25, fps=12.5, weight_dtype="fp16",
              noise_aug_strength=0.0, decode_chunk_size=8, motion_bucket_id=0,
              area=1.05, frame_num=10000, overlap=3, shift_offset=3,
              min_appearance_guidance_scale=2.0, max_appearance_guidance_scale=2.0,
              min_motion_guidance_scale=2.0, max_motion_guidance_scale=2.0,
              i2i_noise_strength=1.0, ip_motion_scale=1.0)
OUTPUT_NAME = "full_resolution.mp4"
MODEL_ID = "tencent/HunyuanPortrait"
BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
ARCFACE_MODEL_ID = "FoivosPar/Arc2Face"
ARCFACE_REVISION = "da2f1e9aa3954dad093213acfc9ae75a68da6ffd"
DETECTOR_MODEL_ID = "LeonJoe13/Sonic"
WEIGHTS = REPO_DIR / "pretrained_weights"
CHECKPOINT_FILES = {
    "unet_checkpoint_path": "unet.pth",
    "pose_guider_checkpoint_path": "pose_guider.pth",
    "dino_checkpoint_path": "dino.pth",
    "image_proj_checkpoint_path": "image_proj.pth",
    "motion_expression_checkpoint_path": "expression.pth",
    "motion_pose_checkpoint_path": "headpose.pth",
    "motion_proj_checkpoint_path": "motion_proj.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--driving", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("--prompt-id must contain only letters, digits, underscores or hyphens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    args.source = args.source.resolve()
    if not args.source.is_file():
        parser.error("--source must point to an existing file")
    args.driving = args.driving.resolve()
    if not args.driving.is_file():
        parser.error("--driving must point to an existing file")
    args.output_dir = args.output_dir.resolve()
    return args


def resolve_weights():
    from huggingface_hub import hf_hub_download, snapshot_download

    base_files = ["scheduler/scheduler_config.json", "unet/config.json", "vae/config.json",
                  "vae/diffusion_pytorch_model.fp16.safetensors"]
    base = WEIGHTS if all((WEIGHTS / name).is_file() for name in base_files) else Path(
        snapshot_download(BASE_MODEL_ID, allow_patterns=base_files))
    paths = {"pretrained_model_name_or_path": str(base)}
    for key, filename in CHECKPOINT_FILES.items():
        local = WEIGHTS / "hyportrait" / filename
        paths[key] = str(local) if local.is_file() else hf_hub_download(MODEL_ID, filename)
    arcface = WEIGHTS / "arcface.onnx"
    paths["arcface_model_path"] = str(arcface) if arcface.is_file() else hf_hub_download(
        ARCFACE_MODEL_ID, "arcface.onnx", revision=ARCFACE_REVISION)
    detector = WEIGHTS / "yoloface_v5m.pt"
    paths["det_path"] = str(detector) if detector.is_file() else hf_hub_download(
        DETECTOR_MODEL_ID, "yoloface_v5m.pt")
    return paths


def generate(args, work):
    import subprocess
    from omegaconf import OmegaConf

    params = dict(PARAMS)
    decode_chunk_size = os.getenv("FAKEPARTS_DECODE_CHUNK_SIZE")
    if decode_chunk_size:
        value = int(decode_chunk_size)
        if not 1 <= value <= PARAMS["n_sample_frames"]:
            raise ValueError("FAKEPARTS_DECODE_CHUNK_SIZE must be between 1 and 25")
        params["decode_chunk_size"] = value
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_PATH), params, resolve_weights(),
                          {"seed": args.seed, "output_dir": str(work / "results")})
    config_path = work / "config.yaml"
    OmegaConf.save(cfg, config_path)
    subprocess.run([sys.executable, str(REPO_DIR / "inference.py"),
                    "--config", str(config_path), "--image_path", str(args.source),
                    "--video_path", str(args.driving)], cwd=REPO_DIR, check=True)
    outputs = list((work / "results").rglob(OUTPUT_NAME))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one {OUTPUT_NAME}, found {len(outputs)}")
    return outputs[0]


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
