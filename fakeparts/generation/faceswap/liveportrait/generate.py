#!/usr/bin/env python3
"""liveportrait repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR.parents[3] / "FakePartsBench/fake/faceswap/liveportrait"
MODEL_ID = "KlingTeam/LivePortrait"
WEIGHTS = REPO_DIR / "pretrained_weights"
CHECKPOINT_FILES = {
    "checkpoint_F": "liveportrait/base_models/appearance_feature_extractor.pth",
    "checkpoint_M": "liveportrait/base_models/motion_extractor.pth",
    "checkpoint_G": "liveportrait/base_models/spade_generator.pth",
    "checkpoint_W": "liveportrait/base_models/warping_module.pth",
    "checkpoint_S": "liveportrait/retargeting_models/stitching_retargeting_module.pth",
}
PARAMS = dict(flag_use_half_precision=True, device_id=0, flag_force_cpu=False,
              flag_crop_driving_video=True, flag_normalize_lip=False,
              flag_source_video_eye_retargeting=False, flag_eye_retargeting=False,
              flag_lip_retargeting=False, flag_stitching=True, flag_relative_motion=True,
              flag_pasteback=True, flag_do_crop=True, driving_option="expression-friendly",
              driving_multiplier=1.0, driving_smooth_observation_variance=3e-7,
              animation_region="all", det_thresh=0.15, scale=2.3,
              vx_ratio=0.0, vy_ratio=-0.125, flag_do_rot=True, source_max_dim=1280,
              source_division=2, scale_crop_driving_video=2.2,
              vx_ratio_crop_driving_video=0.0, vy_ratio_crop_driving_video=-0.1,
              flag_do_torch_compile=False, audio_priority="driving")


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


def load_pipeline(arguments):
    from huggingface_hub import snapshot_download
    from inference import partial_fields
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    required = list(CHECKPOINT_FILES.values()) + ["liveportrait/landmark.onnx"]
    local_ready = all((WEIGHTS / name).is_file() for name in required)
    local_ready = local_ready and bool(list((WEIGHTS / "insightface/models/buffalo_l").glob("*.onnx")))
    weights = WEIGHTS if local_ready else Path(snapshot_download(
        MODEL_ID, allow_patterns=["liveportrait/*", "insightface/*"]))
    inference_cfg = partial_fields(InferenceConfig, vars(arguments))
    for field, filename in CHECKPOINT_FILES.items():
        setattr(inference_cfg, field, str(weights / filename))
    crop_cfg = partial_fields(CropConfig, vars(arguments))
    crop_cfg.insightface_root = str(weights / "insightface")
    crop_cfg.landmark_ckpt_path = str(weights / "liveportrait/landmark.onnx")
    return LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)


def generate(args, work):
    import random
    import numpy as np
    import torch
    from src.config.argument_config import ArgumentConfig
    from inference import fast_check_ffmpeg

    if args.driving.suffix.lower() != ".mp4":
        raise ValueError("--driving must be an MP4 video")
    if args.source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("--source must be a portrait image")
    if not fast_check_ffmpeg():
        raise RuntimeError("FFmpeg must be available on PATH")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # The upstream pipeline writes a motion template beside its driving input.
    driving = work / "driving.mp4"
    shutil.copy2(args.driving, driving)
    arguments = ArgumentConfig(source=str(args.source), driving=str(driving),
                               output_dir=str(work), **PARAMS)
    pipeline = load_pipeline(arguments)
    output, _ = pipeline.execute(arguments)
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
