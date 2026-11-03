#!/usr/bin/env python3
"""opensora adapter using the checked-out official implementation."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_DIR = ADAPTER_DIR
OUTPUT_DIR = ADAPTER_DIR.parents[2] / "FakePartsBench/fake/t2v/opensora"
MODEL_ID = "hpcai-tech/Open-Sora-v2"
CONFIG_PATH = REPO_DIR / "configs/diffusion/inference/256px.py"
SAMPLING = dict(resolution="256px", aspect_ratio="16:9", num_frames=129,
                num_steps=50, shift=True, temporal_reduction=4, is_causal_vae=True,
                guidance=7.5, guidance_img=3.0, text_osci=True, image_osci=True,
                scale_temporal_osci=True, method="i2v")
FPS = 24
DTYPE = "bf16"
MOTION_SCORE = "4"
OFFLOAD = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["t2v", "it2v"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prompt", required=True)
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
    if args.seed == 0:
        parser.error("Open-Sora's upstream sampler treats seed 0 as unset; use a positive seed")
    if args.task == "it2v" and args.input is None:
        parser.error("--input is required for IT2V")
    if args.task == "t2v" and args.input is not None:
        parser.error("--input is only valid for IT2V")
    if args.output_dir == OUTPUT_DIR and args.task == "it2v":
        args.output_dir = OUTPUT_DIR.parents[1] / "it2v/opensora"
    args.output_dir = args.output_dir.resolve()
    return args


def build_config(args, work, checkpoint_root):
    import csv
    from mmengine.config import Config

    cfg = Config.fromfile(str(CONFIG_PATH))
    cfg.sampling_option.update(SAMPLING)
    cfg.sampling_option.seed = args.seed
    cfg.seed = args.seed
    cfg.batch_size = 1
    cfg.num_sample = 1
    cfg.dtype = DTYPE
    cfg.cond_type = "t2v" if args.task == "t2v" else "i2v_head"
    cfg.prompt = None
    prompt_path = work / "prompt.csv"
    with prompt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "ref"] if args.input is not None else ["text"])
        writer.writerow([args.prompt, str(args.input)] if args.input is not None else [args.prompt])
    cfg.dataset.data_path = str(prompt_path)
    cfg.prompt_refine = False
    cfg.offload = OFFLOAD
    cfg.fps_save = FPS
    cfg.motion_score = MOTION_SCORE
    cfg.save_dir = str(work / "results")
    if args.input is not None:
        cfg.ref = str(args.input)
    cfg.model.from_pretrained = str(checkpoint_root / "Open_Sora_v2.safetensors")
    cfg.ae.from_pretrained = str(checkpoint_root / "hunyuan_vae.safetensors")
    cfg.t5.from_pretrained = str(checkpoint_root / "google/t5-v1_1-xxl")
    cfg.clip.from_pretrained = str(checkpoint_root / "openai/clip-vit-large-patch14")
    config_path = work / "config.py"
    cfg.dump(str(config_path))
    return config_path


def generate(args, work):
    import subprocess
    from huggingface_hub import snapshot_download

    checkpoints = REPO_DIR / "ckpts"
    needed = ["Open_Sora_v2.safetensors", "hunyuan_vae.safetensors",
              "google/t5-v1_1-xxl", "openai/clip-vit-large-patch14"]
    if not all((checkpoints / path).exists() for path in needed):
        checkpoints = Path(snapshot_download(MODEL_ID))
    config = build_config(args, work, checkpoints)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=1", str(REPO_DIR / "scripts/diffusion/inference.py"),
                    str(config)], cwd=REPO_DIR, env=env, check=True)
    outputs = list((work / "results").rglob("*.mp4"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one generated video, found {len(outputs)}")
    return outputs[0]


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
