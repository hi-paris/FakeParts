from __future__ import annotations

from argparse import Namespace
import importlib
from pathlib import Path
import sys
from typing import Sequence

from ...base import FrameDetector


class FatFormer(FrameDetector):
    """FatFormer 4-class-supervised CLIP ViT-L/14 image detector."""

    image_size = 224

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != self.image_size:
            raise ValueError(f"FatFormer expects {self.image_size}px crops; got {crop_size}")
        if len(self.ckpt_paths) != 2:
            raise ValueError("FatFormer requires its release checkpoint and CLIP ViT-L/14 weights")

        import torch
        from torchvision import transforms

        self.torch = torch
        self.device_obj = torch.device(device)
        self.vendor_root = Path(__file__).resolve().parent
        optional_deps = Path(__file__).resolve().parents[4] / ".deps" / "detection-extra"
        if optional_deps.is_dir() and str(optional_deps) not in sys.path:
            sys.path.append(str(optional_deps))
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        args = Namespace(
            backbone="CLIP:ViT-L/14", num_classes=2, num_vit_adapter=3,
            num_context_embedding=8, init_context_embedding="",
            hidden_dim=768, clip_vision_width=1024,
            frequency_encoder_layer=2, decoder_layer=4, num_heads=12,
            clip_download_root=str(self.ckpt_paths[1].parent),
        )
        model_factory = importlib.import_module("models").build_model
        self.model = model_factory(args)
        try:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("FatFormer release checkpoint must contain a 'model' state dict")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device_obj).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def predict_frames(self, img_list: Sequence) -> list[dict]:
        if not img_list:
            return []
        batch = self.torch.stack([
            self.transform(image.convert("RGB")) for image in img_list
        ]).to(self.device_obj)
        with self.torch.inference_mode():
            logits = self.model(batch)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"FatFormer expected [batch, 2] logits, got {tuple(logits.shape)}")
            scores = self.torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        return [self.result(score) for score in scores]
