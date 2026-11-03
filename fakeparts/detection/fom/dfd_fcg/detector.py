from __future__ import annotations

import importlib
import math
import pickle
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

from ...base import VideoDetector, is_cuda_oom
from ...utils import cache_video_frames


class DFDFCG(VideoDetector):
    """DFD-FCG full FFG-L/14 video detector."""

    frames_per_clip = 10
    clip_stride_seconds = 0.333
    batch_size = 8

    def set_inference_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, int(batch_size))

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if len(self.ckpt_paths) != 6:
            raise ValueError(
                "DFD-FCG requires setting.yaml, weights.ckpt, face semantic features, "
                "OpenAI CLIP ViT-L/14, 2D-FAN, and SFD weights"
            )

        import torch

        self.torch = torch
        self.device_obj = torch.device(device)
        self.vendor_root = Path(__file__).resolve().parent / "upstream"
        project_root = Path(__file__).resolve().parents[4]
        optional_deps = Path(__file__).resolve().parents[4] / ".deps" / "detection-extra"
        if optional_deps.is_dir() and str(optional_deps) not in sys.path:
            sys.path.append(str(optional_deps))
        detection_deps = project_root / ".deps" / "detection"
        if detection_deps.is_dir() and str(detection_deps) not in sys.path:
            sys.path.append(str(detection_deps))
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        try:
            import face_alignment
        except ImportError as exc:
            raise RuntimeError(
                "DFD-FCG requires the 'face-alignment' package. "
                "Install the project dependencies with `uv sync`."
            ) from exc
        self.face_alignment = face_alignment
        # The upstream project uses absolute ``src.*`` imports. Keep this
        # adapter self-contained by exposing its vendored root for imports.
        import importlib.util
        import types
        src_module = sys.modules.get("src")
        src_paths = list(getattr(src_module, "__path__", ())) if src_module is not None else []
        if str(self.vendor_root / "src") not in src_paths:
            src_module = types.ModuleType("src")
            src_module.__path__ = [str(self.vendor_root / "src")]
            src_module.__package__ = "src"
            sys.modules["src"] = src_module

        import yaml

        setting = yaml.safe_load(self.ckpt_paths[0].read_text(encoding="utf-8"))
        model_config = setting["model"]
        class_path = model_config["class_path"]
        module_name, class_name = class_path.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        feature_path = str(self.ckpt_paths[2])
        clip_root = str(self.ckpt_paths[3].parent)
        self.model = model_class.load_from_checkpoint(
            str(self.ckpt_paths[1]), map_location="cpu", strict=False,
            face_feature_path=feature_path, clip_download_root=clip_root,
        )
        self.model.to(self.device_obj).eval()
        self.transform = self.model.transform
        self.mean_face_path = self.vendor_root / "misc" / "20words_mean_face.npy"
        if not self.mean_face_path.is_file():
            raise FileNotFoundError(f"DFD-FCG mean-face file is missing: {self.mean_face_path}")

    def predict_video(self, video_path):
        with TemporaryDirectory(prefix="fakeparts-dfd-fcg-", dir="/tmp") as work_dir:
            return self.predict_video_with_workdir(video_path, work_dir)

    def predict_video_with_workdir(self, video_path, work_dir):
        import cv2
        import numpy as np

        self.torch.hub.set_dir(str(self.ckpt_paths[4].parent.parent))

        video_path = Path(video_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_data_path = work_dir / "frame_data.pickle"

        # Import the upstream preprocessing functions without running its CLI.
        preprocess = importlib.import_module("src.preprocess.crop_main_face")
        landmark_module = importlib.import_module("src.preprocess.fetch_landmark_bbox")
        dtype = self.torch.float16 if self.device_obj.type == "cuda" else self.torch.float32
        face_landmarker = self.face_alignment.FaceAlignment(
            self.face_alignment.LandmarksType.TWO_D,
            face_detector="sfd", dtype=dtype, flip_input=False,
            face_detector_kwargs={"path_to_detector": str(self.ckpt_paths[5])},
            device=str(self.device_obj),
        )
        cached_dir = cache_video_frames(video_path, self.frame_cache_root) \
            if getattr(self, "frame_cache_root", None) is not None else None
        cached_frames = None
        if cached_dir is not None:
            cached_frames = [
                cv2.imread(str(frame_path))
                for frame_path in sorted(cached_dir.glob("frame_*.jpg"))
            ]
            if not cached_frames or any(frame is None for frame in cached_frames):
                raise RuntimeError(f"DFD-FCG could not read cached frames: {video_path}")
        def detect_landmarks(batch):
            results = face_landmarker.get_landmarks_from_batch(
                batch.to(self.device_obj), return_bboxes=True
            )
            landmarks = [items if items is not None else [] for items in results[0]]
            bboxes = [items if items is not None else [] for items in results[2]]
            return landmarks, results[1], bboxes

        frame_data = landmark_module.landmark_extract(
            fn=detect_landmarks,
            org_path=str(video_path), batch_size=8, max_res=800,
            frames=cached_frames,
        )
        with temp_data_path.open("wb") as stream:
            pickle.dump(frame_data, stream)

        fps, frames = preprocess.get_video_frames(str(video_path), frames=cached_frames)
        if fps <= 0 or not frames:
            raise RuntimeError(f"DFD-FCG could not decode video: {video_path}")
        frame_landmarks, frame_bboxes = preprocess.get_video_frame_data(str(temp_data_path))
        if not any(frame_landmarks):
            print(
                f"[DFD-FCG] No face detected; scoring centered full-frame clips: {video_path}",
                flush=True,
            )
            crop_frames = self._center_crop_frames(frames)
            return self._score_crops(crop_frames, fps, video_path)
        landmarks, bboxes, indices = preprocess.get_main_face_data(
            frame_landmarks, frame_bboxes, d_rate=0.65, max_paddings=fps * 3,
        )
        if len(landmarks) < len(frames) * 0.9:
            print(
                "[DFD-FCG] Main-face track covers fewer than 90% of frames; "
                f"scoring centered full-frame clips ({len(landmarks)}/{len(frames)}): {video_path}",
                flush=True,
            )
            crop_frames = self._center_crop_frames(frames)
            return self._score_crops(crop_frames, fps, video_path)
        crop_frames, _, _ = preprocess.crop_patch(
            frames, landmarks, bboxes, indices,
            np.load(self.mean_face_path), window_margin=12,
            start_idx=15, stop_idx=68, crop_size=150, target_size=256,
        )
        del frames, frame_data, frame_landmarks, frame_bboxes, landmarks, bboxes

        return self._score_crops(crop_frames, fps, video_path)

    def _score_crops(self, crop_frames, fps, video_path):
        import cv2

        offsets = [math.floor(i * self.clip_stride_seconds * fps)
                   for i in range(self.frames_per_clip)]
        clip_count = len(crop_frames) - offsets[-1]
        if clip_count <= 0:
            raise RuntimeError(
                f"DFD-FCG needs at least {offsets[-1] + 1} cropped frames: {video_path}"
            )

        scores = []
        start = 0
        with self.torch.inference_mode():
            while start < clip_count:
                clips = []
                stop = min(start + self.batch_size, clip_count)
                for clip_start in range(start, stop):
                    frames_tensor = []
                    for offset in offsets:
                        rgb = cv2.cvtColor(crop_frames[clip_start + offset], cv2.COLOR_BGR2RGB)
                        frames_tensor.append(self.transform(self._to_pil(rgb)))
                    clips.append(self.torch.stack(frames_tensor))
                batch = self.torch.stack(clips).to(self.device_obj)
                try:
                    logits = self.model.evaluate(batch)["logits"]
                except Exception as error:
                    if not is_cuda_oom(error) or len(clips) <= 1:
                        raise
                    self.batch_size = max(1, len(clips) // 2)
                    self.torch.cuda.empty_cache()
                    print(
                        f"[DFD-FCG] CUDA OOM; reducing clip batch to "
                        f"{self.batch_size}", flush=True,
                    )
                    continue
                scores.extend(self.torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist())
                start = stop
        if not scores:
            raise RuntimeError(f"DFD-FCG produced no clip predictions: {video_path}")
        return self.result(sum(scores) / len(scores))

    @staticmethod
    def _center_crop_frames(frames, target_size=256):
        import cv2

        crop_frames = []
        for frame in frames:
            height, width = frame.shape[:2]
            side = min(height, width)
            top = (height - side) // 2
            left = (width - side) // 2
            square = frame[top:top + side, left:left + side]
            crop_frames.append(cv2.resize(square, (target_size, target_size)))
        return crop_frames

    @staticmethod
    def _to_pil(rgb):
        from PIL import Image
        return Image.fromarray(rgb)
