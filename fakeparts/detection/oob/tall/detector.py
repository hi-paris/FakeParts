from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

from ...base import VideoDetector
from ...utils import cache_video_frames


class TALL(VideoDetector):
    """TALL-Swin video detector using eight deterministic four-frame clips."""

    input_size = 224
    duration = 4
    num_clips = 8
    thumbnail_rows = 2

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != self.input_size:
            raise ValueError(
                f"TALL checkpoint is configured for {self.input_size}px inputs; "
                f"received crop_size={crop_size}"
            )
        import torch
        from timm.models import create_model

        self.torch = torch
        self.device_obj = torch.device(device)
        vendor_root = Path(__file__).resolve().parent
        self.video_dataset = self._load_vendor_module(
            "fakeparts_tall_video_dataset", vendor_root / "video_dataset.py")
        self.video_transforms = self._load_vendor_module(
            "fakeparts_tall_video_transforms", vendor_root / "video_transforms.py")
        self._register_model(vendor_root)

        # These settings match ckpt/detection/TALL/tall.yaml: four frames per
        # thumbnail, 2 rows, 224px resolution, two classes, and hub position
        # embeddings. Eight fixed-offset clips cover the configured 32 frames.
        self.model = create_model(
            "TALL_SWIN", pretrained=False, duration=self.duration,
            hpe_to_token=True, rel_pos=False, window_size=7,
            thumbnail_rows=self.thumbnail_rows, token_mask=True,
            online_learning=False, num_classes=2, drop_rate=0.0,
            drop_path_rate=0.1, drop_block_rate=None, use_checkpoint=False,
        )
        self._load_weights(self.ckpt_path)
        self.model.to(self.device_obj).eval()

    @staticmethod
    def _load_vendor_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load TALL component: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _register_model(cls, vendor_root: Path) -> None:
        package_name = "fakeparts_tall_models"
        if package_name in sys.modules:
            return
        init_file = vendor_root / "my_models" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name, init_file,
            submodule_search_locations=[str(init_file.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load TALL model package: {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    def _load_weights(self, checkpoint_path: Path) -> None:
        try:
            checkpoint = self.torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = self.torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        else:
            raise ValueError("TALL checkpoint must contain a model state dictionary")
        if not isinstance(state, dict):
            raise ValueError("TALL checkpoint model entry is not a state dictionary")
        normalized = {}
        for key, value in state.items():
            key = str(key)
            while key.startswith("module."):
                key = key.removeprefix("module.")
            if key.startswith("model."):
                key = key.removeprefix("model.")
            normalized[key] = value
        state = normalized
        self.model.load_state_dict(state, strict=True)

    def predict_video(self, video_path):
        work_dir = Path(self.work_dir or "/tmp/fakeparts-work") / Path(video_path).stem
        work_dir.mkdir(parents=True, exist_ok=True)
        return self.predict_video_with_workdir(video_path, work_dir)

    def predict_video_with_workdir(self, video_path, work_dir):
        """Decode sampled frames to per-video scratch, then infer and aggregate."""
        video_path = Path(video_path)
        work_dir = Path(work_dir)
        frame_cache_dir = None
        if getattr(self, "frame_cache_root", None) is not None:
            frame_cache_dir = cache_video_frames(video_path, self.frame_cache_root)
            frame_count = len(list(frame_cache_dir.glob("frame_*.jpg")))
        else:
            frame_count = self._count_frames(video_path)
        if frame_count < 3:
            raise RuntimeError(f"TALL requires at least 3 decodable frames: {video_path}")

        indices = self.video_dataset.sample_val_test_clip(
            video_length=frame_count, num_consecutive_frames=1,
            num_frames=self.duration, sample_freq=1, dense_sampling=True,
            fixed_offset=True, num_clips=self.num_clips, whole_video=False,
        )
        selected = set(map(int, indices))
        frames_dir = work_dir / "frames" / "input"
        frames_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        if frame_cache_dir is not None:
            for frame_number in sorted(selected):
                source = frame_cache_dir / f"frame_{frame_number - 1:08d}.jpg"
                if not source.is_file():
                    raise RuntimeError(f"missing cached TALL frame {frame_number}: {source}")
                shutil.copyfile(source, frames_dir / f"{frame_number:04d}.jpg")
                saved += 1
        else:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {video_path}")
            try:
                frame_number = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_number += 1
                    if frame_number in selected:
                        target = frames_dir / f"{frame_number:04d}.jpg"
                        if not cv2.imwrite(str(target), frame):
                            raise RuntimeError(f"failed to stage sampled frame {frame_number}")
                        saved += 1
            finally:
                capture.release()
        if saved != len(selected):
            raise RuntimeError(
                f"video changed or frame count was inconsistent: staged {saved} of "
                f"{len(selected)} sampled frames from {video_path}"
            )

        # This list and the official TALL dataset class preserve the released
        # deterministic dense-sampling and frame ordering behavior.
        list_file = work_dir / "input.txt"
        list_file.write_text(f"frames/input 1 {frame_count} 0\n", encoding="utf-8")
        transform = self._validation_transform()
        dataset = self.video_dataset.VideoDataSet(
            str(work_dir), "input.txt", num_groups=self.duration,
            frames_per_group=1, num_clips=self.num_clips, modality="rgb",
            dense_sampling=True, fixed_offset=True, image_tmpl="{:04d}.jpg",
            transform=transform, is_train=False, test_mode=False,
            seperator=" ", filter_video=3,
        )
        sample, _ = dataset[0]
        sample = sample.unsqueeze(0).to(self.device_obj, non_blocking=True)
        with self.torch.inference_mode():
            logits = self.model(sample)
        if logits.ndim != 2 or logits.shape != (self.num_clips, 2):
            raise ValueError(
                f"TALL expected {self.num_clips} two-class clip logits, got "
                f"{tuple(logits.shape)}"
            )
        # The upstream evaluation averages the clip logits before its video
        # decision; softmax then gives the runner's fake-probability score.
        video_logits = logits.mean(dim=0, keepdim=True)
        fake_probability = self.torch.softmax(video_logits, dim=1)[0, 1].item()
        return self.result(fake_probability)

    def _count_frames(self, video_path: Path) -> int:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        count = 0
        try:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
        finally:
            capture.release()
        return count

    def _validation_transform(self):
        transforms = self.video_transforms
        scaled_size = int(self.input_size / 0.875 + 0.5)
        from torchvision.transforms import Compose

        return Compose([
            transforms.GroupScale(scaled_size),
            transforms.GroupCenterCrop(self.input_size),
            transforms.Stack(),
            transforms.ToTorchFormatTensor(),
            transforms.GroupNormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
