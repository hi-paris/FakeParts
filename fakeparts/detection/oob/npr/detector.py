from __future__ import annotations

import torch
from torchvision import transforms

from ...base import FrameDetector
from .networks.resnet import resnet50


class NPR(FrameDetector):
    """Official NPR ResNet-50 inference adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = resnet50(num_classes=1)
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        # The released NPR checkpoint is a training bundle, while the upstream
        # validation path loads its `model` entry. Some NPR weights are already
        # exported as a plain state dict, so accept both layouts.
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state_dict, dict):
            state_dict = {
                str(key).removeprefix("module."): value
                for key, value in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        crop_size = self.crop_size or 224
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict_frames(self, images):
        inputs = torch.stack([self.transform(image) for image in images]).to(self.device)
        scores = self.model(inputs).sigmoid().flatten().cpu().tolist()
        return [self.result(score) for score in scores]
