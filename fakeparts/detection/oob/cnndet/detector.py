import torch
from torchvision import transforms

from ...base import FrameDetector
from .networks.resnet import resnet50


class CNNDet(FrameDetector):
    """Predict fake scores for batches of RGB PIL images of equal size."""

    input_kind = "frame"
    output_level = "frame"

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        self.model = resnet50(num_classes=1)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()

        preprocessing = []
        if crop_size is not None:
            preprocessing.append(transforms.CenterCrop(crop_size))
        self.transform = transforms.Compose(preprocessing + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def predict_frames(self, img_list):
        inputs = torch.stack([self.transform(image) for image in img_list])
        scores = self.model(inputs.to(self.device)).sigmoid().flatten().cpu().tolist()
        return [self.result(score) for score in scores]
