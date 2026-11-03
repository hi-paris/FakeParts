from __future__ import annotations

from pathlib import Path
import sys

from ...base import FrameDetector


class HiFiNet(FrameDetector):
    """HiFi-Net detection score from the maximum predicted manipulation mask."""

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch
        from torchvision import transforms

        # HiFi's released modules use the upstream package's top-level `models`
        # imports. The detector runs in its own CLI process, so keep that vendor
        # namespace scoped to this process instead of duplicating its network.
        vendor_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(vendor_root))
        try:
            from models.NLCDetection_api import NLCDetection
            from models.seg_hrnet import get_seg_model
            from models.seg_hrnet_config import get_cfg_defaults
        except Exception:
            sys.path.remove(str(vendor_root))
            raise

        feature_net = get_seg_model(get_cfg_defaults())
        segment_net = NLCDetection()
        if len(self.ckpt_paths) != 2:
            raise ValueError("HiFi-Net inference requires its HRNet and NLCDetection checkpoints")
        feature_path, segment_path = self.ckpt_paths
        feature_state = torch.load(feature_path, map_location="cpu", weights_only=False)
        segment_state = torch.load(segment_path, map_location="cpu", weights_only=False)
        feature_state = feature_state.get("model", feature_state)
        segment_state = segment_state.get("model", segment_state)
        if any(str(key).startswith("module.") for key in feature_state):
            feature_state = {str(key).removeprefix("module."): value for key, value in feature_state.items()}
        if any(str(key).startswith("module.") for key in segment_state):
            segment_state = {str(key).removeprefix("module."): value for key, value in segment_state.items()}
        feature_net.load_state_dict(feature_state, strict=True)
        segment_net.load_state_dict(segment_state, strict=True)

        class HiFiPipeline(torch.nn.Module):
            def __init__(self, feature, segment):
                super().__init__()
                self.feature = feature
                self.segment = segment

            def forward(self, images):
                _, mask_binary, _, _, _, final_logits = self.segment(
                    self.feature(images), images)
                fake_probability = 1.0 - final_logits.softmax(dim=1)[:, 0]
                return fake_probability, mask_binary

        self.model = HiFiPipeline(feature_net, segment_net).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict_frames(self, images):
        import torch

        inputs = torch.stack([self.transform(image.convert("RGB")) for image in images])
        with torch.inference_mode():
            scores, _ = self.model(inputs.to(self.device))
            scores = scores.flatten().cpu().tolist()
        return [self.result(score) for score in scores]
