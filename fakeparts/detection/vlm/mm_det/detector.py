from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from fakeparts.detection.base import VideoDetector


class MMDet(VideoDetector):
    """Author-style MM-Det inference, with per-video frames/features under runner /tmp."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.repo = Path(__file__).with_name("runtime")
        sys.path.insert(0, str(self.repo / "LLaVA"))
        sys.path.insert(0, str(self.repo))
        try:
            from models import VectorQuantizedVAE, MMDet as MMDetModel, MMEncoder
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "MM-Det inference needs its optional LLaVA/Transformers and vision dependencies. "
                "Install the dependencies from fakeparts/detection/vlm/mm_det/upstream/requirements.txt "
                "in the model environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("The released MM-Det pipeline requires CUDA for its LLaVA encoder.")
        self.device_obj = torch.device(self.device)
        config = {
            "window_size": 10, "st_pretrained": False, "st_ckpt": None,
            "lmm_ckpt": "sparklexfantasy/llava-7b-1.5-rfrd", "lmm_base": None,
            "load_8bit": False, "load_4bit": False, "conv_mode": "llava_v1",
            "new_tokens": 64, "selected_layers": [-1], "interval": 200,
            "cache_mm": True,
        }
        self.vqvae = VectorQuantizedVAE(3, 256, 512)
        self.vqvae.load_state_dict(torch.load(self.ckpt_paths[1], map_location="cpu", weights_only=False), strict=True)
        self.vqvae = self.vqvae.to(self.device_obj).eval()
        self.mm_encoder = MMEncoder(config).to(self.device_obj).eval()
        self.model = MMDetModel(config).to(self.device_obj).eval()
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.transform = transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.F = torch.nn.functional
        self.window_size = 10
        self.interval = 200

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="fakeparts-mmdet-") as temporary:
            return self.predict_video_with_workdir(video_path, temporary)

    def predict_video_with_workdir(self, video_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
        torch = __import__("torch")
        work_dir = Path(work_dir)
        original_dir = work_dir / "original"
        recons_dir = work_dir / "recons"
        original_dir.mkdir(parents=True, exist_ok=True)
        recons_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        count = 0
        try:
            with torch.inference_mode():
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    original_path = original_dir / f"frame_{count:08d}.jpg"
                    recon_path = recons_dir / f"frame_{count:08d}.jpg"
                    image.save(original_path, quality=95)
                    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)
                    x = (x - 0.5) / 0.5
                    x_recons = self._reconstruct(x.unsqueeze(0).to(self.device_obj))
                    # The authors feed the VQ-VAE [-1, 1] input and save its denormalized output.
                    denorm = (x_recons[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
                    Image.fromarray((denorm.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(recon_path)
                    count += 1
        finally:
            cap.release()
        if count < self.window_size:
            raise ValueError(f"MM-Det requires at least {self.window_size} decoded frames; got {count}.")

        sampled_indices = list(range(0, count, self.interval))
        mm_features: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.inference_mode():
            for frame_idx in sampled_indices:
                image = Image.open(original_dir / f"frame_{frame_idx:08d}.jpg").convert("RGB")
                visual, textual = self.mm_encoder(image)
                mm_features[frame_idx] = (visual.squeeze(0).cpu(), textual.squeeze(0).cpu())
            torch.save(mm_features, work_dir / "mm_features.pth")

            scores = []
            for start in range(0, count - self.window_size + 1, self.window_size):
                orig, recon = [], []
                for frame_idx in range(start, start + self.window_size):
                    frame = Image.open(original_dir / f"frame_{frame_idx:08d}.jpg").convert("RGB")
                    rec = Image.open(recons_dir / f"frame_{frame_idx:08d}.jpg").convert("RGB")
                    orig.append(self.transform(frame))
                    recon.append(self.transform(rec))
                original = torch.stack(orig).unsqueeze(0).to(self.device_obj)
                reconstructed = torch.stack(recon).unsqueeze(0).to(self.device_obj)
                # MM-Det's encoder consumes the first frame of each temporal clip
                # and returns one visual vector plus one generated-token sequence.
                # Keep that batch dimension instead of stacking a feature per frame;
                # the model then fuses those features with the 10-frame backbone output.
                mm_idx = max(index for index in sampled_indices if index <= start)
                cached = {
                    "visual": mm_features[mm_idx][0].unsqueeze(0).to(self.device_obj),
                    "textual": mm_features[mm_idx][1].unsqueeze(0).to(self.device_obj),
                }
                logits = self.model((original, reconstructed), cached_features=cached)
                scores.append(float(logits.softmax(dim=-1)[0, 1].item()))
        score = sum(scores) / len(scores)
        return self.result(score) | {"frames": count, "windows": len(scores),
                                     "multimodal_feature_frames": len(mm_features)}

    def _reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        recons, _, _ = self.vqvae(image)
        if recons.shape != image.shape:
            recons = self.F.interpolate(recons, size=image.shape[-2:], mode="nearest")
        return recons
