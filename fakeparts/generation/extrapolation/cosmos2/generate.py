#!/usr/bin/env python3
"""
Diffusers Video-to-Video Extrapolation using CosmosPredict 2 model.

Video params (new extrapolated clip):
- Frames: 129 (with 40 input condition frames and 89 generated frames)
- FPS: 25
- Length: 5s
- Guidance Scale: 7.0
- Inference Steps: 35 (legacy Diffusers script configuration)
"""

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
MODEL_ID = "nvidia/Cosmos-Predict2-14B-Video2World"
HEIGHT = 704
WIDTH = 1280
NUM_FRAMES = 129
CONDITION_FRAMES = 40
FPS = 25
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 35
DTYPE = "bfloat16"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "FakePartsBench/fake/extrapolation/cosmos2"
NEG_PROMPT = (
    "jitter, flicker, sudden scene jump, heavy blur, ghosting, compression artifacts, "
    "low resolution, pixelization on any non-sexual content, pixelization on faces"
)
BASE_PROMPT = (
    "Continue the input video realistically and smoothly, following its objects' features, "
    "motion and scene. Avoid pixelization, blurriness, flicker, ghosting, compression "
    "artifacts, or low resolution. Keep the same style, colour tone, and quality as the input video."
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CosmosSafetySkipper(nn.Module):
    """
    SHOULD NOT MODIFY OR REMOVE
    No-op safety checker compatible with Cosmos2VideoToWorldPipeline.
    - Provides .dtype/.device via a registered buffer so .to(device, dtype) works.
    - Implements check_text_safety(), check_image_safety(), and forward().
    - Always passes content through unchanged; reports 'blocked=False'.
    """

    def __init__(self, log: bool = True) -> None:
        super().__init__()
        self.log = log
        self.register_buffer("_compat_buf", torch.zeros((), dtype=torch.float32), persistent=False)

    @property
    def dtype(self):
        return self._compat_buf.dtype

    @property
    def device(self):
        return self._compat_buf.device

    # ---- Text safety hook expected by the Cosmos pipeline ----
    @torch.no_grad()
    def check_text_safety(self, text: Union[str, List[str]], **kwargs: Any) -> bool:
        # Return True => safe. Accept both str and list[str].
        return True

    # ---- Image safety hook (diffusers style) ----
    @torch.no_grad()
    def check_image_safety(
            self,
            images: TensorOrImages,
            **kwargs: Any
    ) -> Tuple[TensorOrImages, List[Dict[str, Any]]]:
        if isinstance(images, torch.Tensor):
            b = images.shape[0] if images.ndim >= 4 else 1
        else:
            b = len(images)
        reports = [dict(blocked=False, reason=None, score=0.0) for _ in range(b)]
        if self.log:
            print(f"[CosmosSafetySkipper] image safety skipped, batch={b}")
        return images, reports

    @torch.no_grad()
    def check_video_safety(self, video, **kwargs):
        """
        No-op video hook expected by Cosmos2VideoToWorldPipeline.
        Must return the same type that the pipeline passed in (e.g., list[PIL.Image]
        or a Tensor of frames). We just pass it through unchanged.
        """
        if self.log:
            try:
                # best-effort shape/len logging without assuming a type
                if hasattr(video, "shape"):
                    print(f"[CosmosSafetySkipper] video safety skipped, shape={tuple(video.shape)}")
                else:
                    print(f"[CosmosSafetySkipper] video safety skipped, len={len(video)}")
            except Exception:
                print("[CosmosSafetySkipper] video safety skipped.")
        return video

    # ---- Make it also behave like the usual diffusers safety checker ----
    @torch.no_grad()
    def forward(self, images: TensorOrImages, **kwargs: Any):
        return self.check_image_safety(images, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Cosmos Predict2 video conditioned on an image or video."
    )
    parser.add_argument("--input", type=Path, required=True, help="Conditioning image or video path.")
    parser.add_argument("--prompt-id", required=True, help="ID of the prompt used for generation.")
    parser.add_argument("--prompt", default="", help="Optional source caption, prepended to the continuation prompt.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for generation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory in which to save the generated video.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prompt_id):
        parser.error("Invalid --prompt-id")
    if not args.prompt.strip():
        parser.error("--prompt must not be blank")
    if not 0 <= args.seed < 2 ** 63:
        parser.error("--seed must be in [0, 2**63)")
    if not args.input.is_file():
        parser.error("--input must point to an existing file")
    return args


def prepare_conditioning(path):
    from diffusers.utils import load_image, load_video
    from PIL import Image, ImageOps

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frames = [load_image(str(path))]
    else:
        # Match the legacy V2V pipeline: extrapolate from the end of the source clip.
        frames = load_video(str(path))[-CONDITION_FRAMES:]
        if len(frames) < CONDITION_FRAMES:
            raise ValueError(f"Video must contain at least {CONDITION_FRAMES} frames")
    return [ImageOps.fit(frame.convert("RGB"), (WIDTH, HEIGHT),
                         method=Image.Resampling.LANCZOS) for frame in frames]


def load_pipeline():
    import torch
    from diffusers import Cosmos2VideoToWorldPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Cosmos requires CUDA")
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=getattr(torch, DTYPE),
        safety_checker=CosmosSafetySkipper  # must apply to pipeline
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def generate(pipe, prompt, seed, frames):
    import torch

    caption = (prompt or "").strip()
    prompt = f"{caption}. {BASE_PROMPT}" if caption else BASE_PROMPT
    conditioning = {"image": frames[0]} if len(frames) == 1 else {"video": frames}
    output = pipe(
        **conditioning,
        prompt=prompt, negative_prompt=NEG_PROMPT, height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES, fps=FPS, guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).frames[0]
    if len(output) != NUM_FRAMES:
        raise RuntimeError(f"Expected {NUM_FRAMES} frames, received {len(output)}")
    # Preserve the original ending in the conditioned prefix, as in the legacy pipeline.
    return frames + list(output[len(frames):])


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / f"{args.prompt_id}_{args.seed}.mp4"
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    from diffusers.utils import export_to_video

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditioning = prepare_conditioning(args.input)
    frames = generate(load_pipeline(), args.prompt, args.seed, conditioning)
    with tempfile.TemporaryDirectory(prefix=".generate-", dir=args.output_dir) as temp:
        staged = Path(temp) / "result.mp4"
        export_to_video(frames, str(staged), fps=FPS)
        os.link(staged, output_path)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
