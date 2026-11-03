from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Sequence

from ...base import FrameDetector


class IMLViT(FrameDetector):
    """IML-ViT localization mask converted to a frame-level anomaly score."""

    input_size = 1024

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != self.input_size:
            raise ValueError(
                f"IML-ViT uses {self.input_size}px padded inputs; got crop_size={crop_size}"
            )
        import torch

        self.torch = torch
        self.device_obj = torch.device(device)
        self.vendor_root = Path(__file__).resolve().parent
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        model_module = importlib.import_module("iml_vit_model")
        self.model = model_module.iml_vit_model(input_size=self.input_size)
        try:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError("IML-ViT checkpoint must contain a model state dictionary")
        checkpoint = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
        }
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device_obj).eval()
        transforms = importlib.import_module("utils.iml_transforms")
        self.pad_transform = transforms.get_albu_transforms(
            type_="pad", outputsize=self.input_size)

    def predict_frames(self, img_list: Sequence) -> list[dict]:
        if not img_list:
            return []
        import numpy as np

        tensors = []
        for image in img_list:
            if hasattr(image, "convert"):
                image = np.asarray(image.convert("RGB"))
            else:
                image = np.asarray(image)
                if image.ndim == 3 and image.shape[2] == 3:
                    # FrameDetector inputs are RGB; OpenCV frames are converted by the runner.
                    pass
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"IML-ViT expects RGB images, got {image.shape}")
            transformed = self.pad_transform(image=image)
            tensors.append(transformed["image"])
        batch = self.torch.stack(tensors).to(self.device_obj)
        with self.torch.inference_mode():
            features = self.model.encoder_net(batch)
            pyramid = self.model.featurePyramid_net(features)
            logits = self.model.predict_head(list(pyramid.values()))
            mask = self.torch.nn.functional.interpolate(
                logits, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            ).sigmoid()
            # The official model emits manipulation masks rather than image labels.
            # Use the strongest localized response as the frame-level detection score.
            scores = mask.flatten(1).amax(dim=1).cpu().tolist()
        return [self.result(score) for score in scores]
