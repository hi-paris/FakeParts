#!/usr/bin/env python3
"""anyv2v repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR.parents[2] / "FakePartsBench/fake/style/anyv2v"
BACKEND = REPO_DIR / "i2vgen-xl"
DEVICE = "cuda:0"
IMAGE_SIZE = [512, 512]
N_FRAMES = 16
FPS = 8
INVERSION_STEPS = 500
EDIT_STEPS = 50
INVERSION_GUIDANCE = 1.0
EDIT_GUIDANCE = 9.0
DDIM_INIT_INDEX = 1
PNP_FEATURE = 0.2
PNP_SPATIAL = 0.2
PNP_TEMPORAL = 0.5
RANDOM_RATIO = 0.0
NEGATIVE_PROMPT = "Distorted motion, blur, artifacts, malformed anatomy."
STRENGTH_CONFIGS = {
    "0.3": {"ddim_init_latents_t_idx": 9, "cfg": 3.4},
    "0.7": {"ddim_init_latents_t_idx": 1, "cfg": 6.6},
    "1.0": {"ddim_init_latents_t_idx": 0, "cfg": 9.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--edited-first-frame", type=Path)
    parser.add_argument("--edit-prompt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", choices=tuple(STRENGTH_CONFIGS), default="0.7")
    parser.add_argument("--output-name")
    parser.add_argument("--device", default=DEVICE)
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
    if args.edited_first_frame is not None:
        args.edited_first_frame = args.edited_first_frame.resolve()
        if not args.edited_first_frame.is_file():
            parser.error("--edited-first-frame must point to an existing file")
    elif not args.edit_prompt or not args.edit_prompt.strip():
        parser.error("provide --edited-first-frame or --edit-prompt")
    args.output_dir = args.output_dir.resolve()
    if args.output_name is not None and (
        Path(args.output_name).name != args.output_name
        or not args.output_name.lower().endswith(".mp4")
    ):
        parser.error("--output-name must be a plain .mp4 filename")
    return args


def generate(args, work):
    import cv2
    import json
    import subprocess
    from omegaconf import OmegaConf
    from PIL import Image

    if args.input.suffix.lower() != ".mp4":
        raise ValueError("--input must be an MP4 video")
    capture = cv2.VideoCapture(str(args.input))
    first_frame = None
    try:
        for frame_index in range(N_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"AnyV2V requires at least {N_FRAMES} decodable frames")
            if frame_index == 0:
                first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if first_frame is None:
        raise ValueError("AnyV2V input has no decodable first frame")
    full_edit = args.edited_first_frame
    if full_edit is None:
        from black_box_image_edit import CosXLEdit, InstructPix2Pix, MagicBrush

        editors = {
            "instructpix2pix": InstructPix2Pix,
            "magicbrush": MagicBrush,
            "cosxl": CosXLEdit,
        }
        editor = editors[args.image_model](device=args.device)
        edited = editor.infer_one_image(
            first_frame.resize((512, 512)), instruct_prompt=args.edit_prompt,
            seed=args.seed, negative_prompt=NEGATIVE_PROMPT,
        ).resize(first_frame.size)
        full_edit = work / "full_strength_first_frame.png"
        edited.save(full_edit)
    else:
        with Image.open(full_edit) as edited_image:
            edited = edited_image.convert("RGB").resize(first_frame.size)
    alpha = float(args.strength)
    blended_frame = Image.blend(first_frame, edited, alpha)
    edited_first_frame = work / "edited_first_frame.png"
    blended_frame.save(edited_first_frame)
    # Use a staged input because upstream frame extraction writes beside the video.
    video = work / "input.mp4"
    shutil.copy2(args.input, video)
    inversion = OmegaConf.load(BACKEND / "configs/group_ddim_inversion/template.yaml")
    editing = OmegaConf.load(BACKEND / "configs/group_pnp_edit/template.yaml")
    common = dict(seed=args.seed, device=args.device, debug=False, data_dir=str(work),
                  video_dir=str(work), video_name="input", video_path=str(video),
                  video_frames_path=str(work / "input"), image_size=IMAGE_SIZE,
                  n_frames=N_FRAMES)
    inversion = OmegaConf.merge(inversion, common, {
        "output_dir": str(work / "inversion"), "force_recompute_latents": True,
        "inverse_config": {"image_size": IMAGE_SIZE, "n_frames": N_FRAMES,
            "cfg": INVERSION_GUIDANCE, "target_fps": FPS, "prompt": "",
            "negative_prompt": "", "n_steps": INVERSION_STEPS,
            "output_dir": str(work / "latents"), "inverse_static_video": False,
            "null_image_inversion": False},
        "recon_config": {"enable_recon": False}})
    strength = STRENGTH_CONFIGS[args.strength]
    editing = OmegaConf.merge(editing, common, {
        "output_dir": str(work / "edited"), "edited_video_name": "output",
        "edited_first_frame_path": str(edited_first_frame),
        "ddim_latents_path": str(work / "latents"), "cfg": strength["cfg"],
        "target_fps": FPS, "editing_prompt": args.prompt,
        "editing_negative_prompt": NEGATIVE_PROMPT, "n_steps": EDIT_STEPS,
        "ddim_init_latents_t_idx": strength["ddim_init_latents_t_idx"], "ddim_inv_prompt": "",
        "random_ratio": RANDOM_RATIO, "pnp_f_t": 1.0,
        "pnp_spatial_attn_t": 1.0, "pnp_temp_attn_t": 1.0})
    group_path = work / "group.json"
    group_path.write_text(json.dumps([{"active": True}]), encoding="utf-8")
    for name, config in (("ddim_inversion", inversion), ("pnp_edit", editing)):
        config_path = work / (name + ".yaml")
        OmegaConf.save(config, config_path)
        subprocess.run([sys.executable, str(BACKEND / ("run_group_" + name + ".py")),
                        "--template_config", str(config_path),
                        "--configs_json", str(group_path)], cwd=BACKEND, check=True)
    outputs = list((work / "edited").rglob("video.mp4"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one edited video, found {len(outputs)}")
    return outputs[0]


def main():
    args = parse_args()
    output = args.output_dir / (args.output_name or f"{args.prompt_id}_{args.seed}.mp4")
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
