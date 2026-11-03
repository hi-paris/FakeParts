from __future__ import annotations

from pathlib import Path

from ...base import FrameDetector


class UnivFD(FrameDetector):
    """UniversalFakeDetect's frozen CLIP backbone and released FC weights."""

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch
        from torchvision import transforms

        from .models import get_model

        clip_cache = (
            self.ckpt_paths[1] if len(self.ckpt_paths) > 1 else
            Path.home() / ".cache" / "clip" / "ViT-L-14.pt"
        )
        if not clip_cache.is_file():
            raise FileNotFoundError(
                "UnivFD needs the OpenAI CLIP ViT-L/14 backbone in "
                f"{clip_cache}; the adapter will not download it automatically"
            )
        self.model = get_model("CLIP:ViT-L/14", clip_download_root=str(clip_cache.parent))
        state = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.fc.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size or 224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def predict_frames(self, images):
        import torch

        inputs = torch.stack([self.transform(image.convert("RGB")) for image in images])
        with torch.inference_mode():
            scores = self.model(inputs.to(self.device)).sigmoid().flatten().cpu().tolist()
        return [self.result(score) for score in scores]
