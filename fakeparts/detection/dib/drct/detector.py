from __future__ import annotations

from pathlib import Path
import sys

from ...base import FrameDetector


class DRCT(FrameDetector):
    """DRCT-2M / GenImage frame classifier with explicit checkpoint variants."""

    ARCHITECTURES = {
        "drct2m_sdv14_convnext": ("convnext_base_in22k", 1024),
        "drct2m_sdv2_convnext": ("convnext_base_in22k", 1024),
        "genimage_sdv14_convnext": ("convnext_base_in22k", 1024),
        "drct2m_sdv14_clip_vitl14": ("clip-ViT-L-14", 1024),
        "drct2m_sdv2_clip_vitl14": ("clip-ViT-L-14", 1024),
        "genimage_sdv14_clip_vitl14": ("clip-ViT-L-14", 1024),
    }

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch

        try:
            model_name, embedding_size = self.ARCHITECTURES[version]
        except KeyError as error:
            raise ValueError(f"Unsupported DRCT version {version!r}") from error

        if "clip" in model_name.lower():
            clip_cache = Path.home() / ".cache" / "clip" / "ViT-L-14.pt"
            if not clip_cache.is_file():
                raise FileNotFoundError(
                    "This DRCT variant needs the OpenAI CLIP ViT-L/14 backbone in "
                    f"{clip_cache}; the adapter will not download it automatically"
                )

        vendor_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(vendor_root))
        try:
            from network.models import get_models
            from data.transform import create_val_transforms
        except Exception:
            sys.path.remove(str(vendor_root))
            raise

        self.model = get_models(
            model_name=model_name,
            pretrained=False,
            num_classes=2,
            embedding_size=embedding_size,
            freeze_extractor=False,
        )
        state = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(str(key).startswith("module.") for key in state):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.transform = create_val_transforms(size=crop_size or 224, is_crop=True)

    def predict_frames(self, images):
        import numpy as np
        import torch

        tensors = [
            self.transform(image=np.asarray(image.convert("RGB")))["image"]
            for image in images
        ]
        inputs = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            scores = self.model(inputs).float().softmax(dim=1)[:, 1].cpu().tolist()
        return [self.result(score) for score in scores]
