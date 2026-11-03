from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fakeparts.detection.base import VideoDetector


class NSGVD(VideoDetector):
    """Single-video adapters for NSG-VD dMMD and its four released baselines."""

    def __init__(self, *args: Any, reference_dir: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.repo = Path(__file__).parent
        self._load_model()

    def _load_model(self) -> None:
        import torch

        self.torch = torch
        sys.path.insert(0, str(self.repo))
        torch_device = torch.device(self.device)
        if self.version.startswith("baseline_"):
            self._load_baseline(torch_device)
        else:
            self._load_mmd(torch_device)

    def _load_baseline(self, device: Any) -> None:
        import torch
        from torchvision import transforms

        _, task, generator, family = self.version.split("_", 3)
        del task, generator  # They identify the fine-tuning recipe in the checkpoint filename.
        from models.demamba import XCLIP_DeMamba
        from models.npr import resnet50_npr
        from models.stil import Det_STIL
        from models.tall import TALL_SWIN

        if family == "demamba":
            model = XCLIP_DeMamba()
        elif family == "npr":
            model = resnet50_npr(pretrained=False)
        elif family == "stil":
            model = Det_STIL(pretrained=False)
        elif family == "tall":
            model = TALL_SWIN(pretrained=False)
        else:
            raise ValueError(f"Unknown NSG-VD baseline {family!r}")
        checkpoint = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        cleaned = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        model.load_state_dict(cleaned, strict=True)
        self.model = model.to(device).eval()
        self.device_obj = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.family = family

    def _sample_frames(self, video_path: Path, count: int, work_dir: Path) -> list[Path]:
        frame_cache_root = getattr(self, "frame_cache_root", None)
        if frame_cache_root is not None:
            from fakeparts.detection.utils import cache_video_frames

            frame_dir = cache_video_frames(video_path, frame_cache_root)
            cached_paths = sorted(frame_dir.glob("frame_*.jpg"))
            if not cached_paths:
                raise RuntimeError(f"No cached frames found for {video_path}")
            indices = np.linspace(
                0, len(cached_paths) - 1, min(count, len(cached_paths)), dtype=int
            )
            outputs = [cached_paths[int(index)] for index in indices]
            while outputs and len(outputs) < count:
                outputs.append(outputs[-1])
            return outputs

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        indices = np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int)
        outputs = []
        for output_index, target in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok:
                continue
            path = work_dir / f"sample_{output_index:03d}.jpg"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
            outputs.append(path)
        cap.release()
        if len(outputs) < count:
            # The reference implementation assumes eight frames. Repeat the final sampled frame
            # for short clips, keeping its expected temporal tensor shape.
            while outputs and len(outputs) < count:
                outputs.append(outputs[-1])
        return outputs

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="fakeparts-nsg-") as temporary:
            return self.predict_video_with_workdir(video_path, temporary)

    def predict_video_with_workdir(self, video_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
        if self.version.startswith("baseline_"):
            return self._predict_baseline(Path(video_path), Path(work_dir))
        return self._predict_mmd(Path(video_path), Path(work_dir))

    def _predict_baseline(self, video_path: Path, work_dir: Path) -> dict[str, Any]:
        torch = self.torch
        frame_paths = self._sample_frames(video_path, 8, work_dir)
        frames = torch.stack([self.transform(Image.open(path).convert("RGB")) for path in frame_paths])
        with torch.inference_mode():
            logits = self.model(frames.unsqueeze(0).to(self.device_obj)).reshape(1, -1)
            score = float(logits[0, 0].sigmoid().item())
        return self.result(score) | {"classifier": self.family, "frames": len(frame_paths)}

    def _make_score_features(self, video_path: Path, work_dir: Path, cache_name: str):
        import torch
        import torch.nn.functional as F
        from data.utils import get_score_fn

        frames_dir = work_dir / cache_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        paths = self._sample_frames(video_path, 8, frames_dir)
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            images.append(torch.from_numpy(np.asarray(image).copy()).float().div(255).permute(2, 0, 1))
        image_tensor = torch.stack(images).unsqueeze(0).to(self.device_obj)
        resized = F.interpolate(image_tensor.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False)
        if self.score_fn is None:
            config_path = self.repo / "libs/eps_ad/imagenet.yml"
            self.score_fn = get_score_fn(
                self.device_obj,
                args_path=str(self.repo / "libs/eps_ad/args.yml"),
                config_path=str(self.repo / "libs/eps_ad/imagenet.yml"),
                process_shape=(3, 256, 256),
                diffusion_checkpoint=str(self.ckpt_paths[0]),
            )
        t = torch.tensor(6 / 1000, device=self.device_obj)
        score = self.score_fn(2 * resized - 1, t.expand(resized.shape[0]))
        score, _ = torch.split(score, score.shape[1] // 2, dim=1)
        score = F.interpolate(score, size=(224, 224), mode="bilinear", align_corners=False)
        score = score.reshape(1, 8, *score.shape[1:])
        image = image_tensor
        pixel_diff = torch.zeros_like(image)
        pixel_diff[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        pixel_diff[:, 0] = image[:, 1] - image[:, 0]
        pixel_diff[:, -1] = image[:, -1] - image[:, -2]
        velocity = score / ((score * pixel_diff).sum((2, 3, 4), keepdim=True) + 1e-10)
        np.save(frames_dir / "score.npy", score[0].detach().float().cpu().numpy())
        np.save(frames_dir / "nsg.npy", velocity[0].detach().float().cpu().numpy())
        return score, velocity, image_tensor

    def _load_mmd(self, device: Any) -> None:
        import torch
        from models.deep_mmd import deep_MMD
        from models.tall import SingleSwinBlockDiscriminator

        _, generator, kind = self.version.split("_")
        config_path = self.repo / "configs/nsg-vd-224x224/standard.yaml"
        is_mp = kind == "mp"
        discriminator = SingleSwinBlockDiscriminator(num_features=300)
        self.model = deep_MMD(discriminator, sigma=1000, sigma0=0.1, epsilon=10,
                              img_size=224, is_yy_zero=is_mp, is_smooth=True)
        state = torch.load(self.ckpt_paths[-1], map_location=device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()
        self.device_obj = device
        self.score_fn = None
        self.mmd_kind = kind
        self.generator = generator
        self.reference_cache = None
        if self.reference_dir is None:
            raise ValueError(
                "NSG-VD dMMD requires the author's real reference distribution. "
                "Pass reference_dir=... (for example the MSR-VTT validation videos)."
            )

    def _predict_mmd(self, video_path: Path, work_dir: Path) -> dict[str, Any]:
        # Every per-reference diffusion feature is cached for subsequent
        # videos. Keep all intermediate tensors out of autograd so the 100
        # reference clips do not retain one diffusion graph each on the GPU.
        with self.torch.inference_mode():
            return self._predict_mmd_inference(video_path, work_dir)

    def _predict_mmd_inference(self, video_path: Path, work_dir: Path) -> dict[str, Any]:
        import torch
        from utils.mmd_utils import MMD_batch2

        _, velocity, image = self._make_score_features(video_path, work_dir, "sampled")
        reference_scores = []
        reference_images = []
        if self.reference_cache is None:
            ref_videos = sorted(p for p in self.reference_dir.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
            if not ref_videos:
                raise ValueError(f"No reference videos found under {self.reference_dir}")
            ref_dir = work_dir / "reference"
            ref_dir.mkdir()
            for idx, ref_video in enumerate(ref_videos[:100]):
                _, ref_velocity, ref_image = self._make_score_features(ref_video, ref_dir / str(idx), f"ref_{idx}")
                reference_scores.append(ref_velocity)
                reference_images.append(ref_image)
            ref_data = torch.cat(reference_images, dim=0).to(self.device_obj)
            ref_features = torch.cat(reference_scores, dim=0).to(self.device_obj)
            self.reference_cache = (ref_features, ref_data, len(reference_scores))
        else:
            ref_features, ref_data, reference_count = self.reference_cache
            reference_scores = [None] * reference_count
        _, feature_test = self.model.net(velocity, out_feature=True)
        _, feature_ref = self.model.net(ref_features, out_feature=True)
        raw = MMD_batch2(torch.cat([feature_ref, feature_test]), feature_ref.shape[0],
                         torch.cat([ref_data, image.to(self.device_obj)]).flatten(1),
                         self.model.sigma, self.model.sigma0_u, self.model.ep,
                         is_smooth=self.model.is_smooth)
        # Original evaluator reports raw dMMD and thresholds at 1 for binary evaluation.
        score = float(torch.sigmoid(raw.reshape(-1)[-1] - 1).item())
        return self.result(score) | {"raw_mmd": float(raw.reshape(-1)[-1].item()),
                                     "mmd_type": self.mmd_kind, "reference_videos": len(reference_scores)}
