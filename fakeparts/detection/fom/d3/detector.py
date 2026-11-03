from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Sequence

from ...base import FrameDetector


class D3(FrameDetector):
    """D³ image detector using the released CLIP ViT-L/14 attention head."""

    image_size = 224

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != self.image_size:
            raise ValueError(f"D³ requires {self.image_size}px inputs; got crop_size={crop_size}")
        if len(self.ckpt_paths) != 2:
            raise ValueError("D³ requires its released attention head and CLIP ViT-L/14 weights")

        import torch
        from PIL import Image
        from torchvision import transforms

        self.torch = torch
        self.Image = Image
        self.device_obj = torch.device(device)
        self.vendor_root = Path(__file__).resolve().parent
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))

        clip_models = importlib.import_module("models.clip_models")
        self.model = clip_models.CLIPModelShuffleAttentionPenultimateLayer(
            "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[14],
            clip_download_root=str(self.ckpt_paths[1].parent),
        )
        try:
            state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("D³ classifier checkpoint must contain an attention-head state dict")
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        self.model.attention_head.load_state_dict(state, strict=True)
        self.model.to(self.device_obj).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def predict_frames(self, img_list: Sequence) -> list[dict]:
        if not img_list:
            return []
        batch = self.torch.stack([
            self.transform(image.convert("RGB") if hasattr(image, "convert") else
                           self.Image.fromarray(image).convert("RGB"))
            for image in img_list
        ]).to(self.device_obj)
        with self.torch.inference_mode():
            logits = self.model(batch).reshape(-1)
            scores = self.torch.sigmoid(logits).cpu().tolist()
        if len(scores) != len(img_list):
            raise ValueError(f"D³ returned {len(scores)} scores for {len(img_list)} frames")
        return [self.result(score) for score in scores]
