#!/usr/bin/env python3
"""insightface repository adapter. Run in this repository's own dependency environment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO_DIR = Path(__file__).resolve().parent
ROOT = REPO_DIR.parents[3]
OUTPUT_DIR = REPO_DIR.parents[2] / "FakePartsBench/fake/faceswap/insightface"
MODEL_NAME = "buffalo_l"
WEIGHTS = ROOT / "ckpt/generation/InsightFace"
LEGACY_WEIGHTS = ROOT.parent / "FakePartsBench/ckpt/generation/InsightFace"
SWAP_MODEL = WEIGHTS / "inswapper_128.onnx"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DET_SIZE = (640, 640)
DET_THRESH = 0.5
CTX_ID = 0
# Largest source face supplies identity; all detected target faces are replaced.
FOURCC = "mp4v"
REQUIRED_MODELS = (
    "inswapper_128.onnx",
    "models/buffalo_l/1k3d68.onnx",
    "models/buffalo_l/2d106det.onnx",
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/genderage.onnx",
    "models/buffalo_l/w600k_r50.onnx",
)


def has_required_models(root):
    return all((root / relative).is_file() for relative in REQUIRED_MODELS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
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
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error("--input must point to an existing file")
    args.output_dir = args.output_dir.resolve()
    return args


def load_pipeline():
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    weights = WEIGHTS
    if not has_required_models(weights) and has_required_models(LEGACY_WEIGHTS):
        weights = LEGACY_WEIGHTS
    swap_model = weights / "inswapper_128.onnx"
    if not swap_model.is_file():
        raise FileNotFoundError(
            f"InsightFace weights are missing. Run `bash scripts/set_up_generator.sh "
            f"--methods insightface` (or `sbatch scripts/set_up_generator.sh "
            f"--methods insightface`). Expected {WEIGHTS}"
        )
    app = FaceAnalysis(name=MODEL_NAME, root=str(weights), providers=PROVIDERS)
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
    swapper = get_model(str(swap_model), providers=PROVIDERS)
    return app, swapper


def generate(args, work):
    import cv2
    import math
    import numpy as np

    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed % (2**31))
    source = cv2.imread(str(args.source))
    if source is None:
        raise ValueError("Cannot decode source image")
    app, swapper = load_pipeline()
    faces = app.get(source)
    if not faces:
        raise ValueError("No face detected in source image")
    identity = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    cap = cv2.VideoCapture(str(args.input))
    writer = None
    output = work / "result.mp4"
    count = swapped = 0
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open input video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has an invalid FPS")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                if width % 2 or height % 2:
                    raise ValueError("Input dimensions must be even for MP4 encoding")
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*FOURCC),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Cannot initialize MP4 writer")
            targets = app.get(frame)
            for face in targets:
                frame = swapper.get(frame, face, identity, paste_back=True)
                swapped += 1
            writer.write(frame)
            count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not count or not swapped:
        raise ValueError("No video frames with detectable target faces were processed")
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
