from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

from ...base import VideoDetector, is_cuda_oom
from ...utils import cache_video_frames


class TFCU(VideoDetector):
    """TFCU video inference with per-video face/landmark preparation."""

    image_size = 224
    frames_per_clip = 4
    feature_batch_size = 16

    def set_inference_batch_size(self, batch_size: int) -> None:
        self.feature_batch_size = max(1, int(batch_size))

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        if crop_size is not None and crop_size != self.image_size:
            raise ValueError(
                f"TFCU checkpoint requires {self.image_size}px faces; "
                f"received crop_size={crop_size}"
            )
        if len(self.ckpt_paths) != 2:
            raise ValueError("TFCU requires the Final.tar model and RetinaFace checkpoints")

        import torch
        import numpy as np

        self.torch = torch
        self.device_obj = torch.device(device)
        self.vendor_root = Path(__file__).resolve().parent
        vendor_root = str(self.vendor_root)
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)

        model_module = importlib.import_module("models.TFCU")
        self.model = model_module.TFCU_Model()
        self._load_weights(self.ckpt_path)
        self.model.to(self.device_obj).eval()

        detector_module = importlib.import_module("lib.ct.detection")
        gpu_id = self.device_obj.index
        if self.device_obj.type == "cuda" and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.face_detector = detector_module.FaceDetector(
            gpu_id=-1 if self.device_obj.type == "cpu" else gpu_id,
            model_path=str(self.ckpt_paths[1]),
        )
        import dlib

        self.landmark_detector = dlib.get_frontal_face_detector()
        predictor_path = self.vendor_root / "lib" / "shape_predictor_81_face_landmarks.dat"
        if not predictor_path.is_file():
            raise FileNotFoundError(f"TFCU's 81-point landmark model is missing: {predictor_path}")
        self.landmark_predictor = dlib.shape_predictor(str(predictor_path))
        self.face_alignment = self._load_module(
            "fakeparts_tfcu_face_utils", self.vendor_root / "face_utils.py")
        default_landmarks = np.load(self.vendor_root / "datasets" / "defalut.npy")
        self.default_landmarks = default_landmarks[0]

    @staticmethod
    def _load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load TFCU component: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def _load_weights(self, checkpoint_path: Path) -> None:
        try:
            checkpoint = self.torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = self.torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            raise ValueError("TFCU checkpoint must contain a model state dictionary")
        if not isinstance(state, dict):
            raise ValueError("TFCU checkpoint state is not a dictionary")
        state = {
            str(key).removeprefix("module."): value
            for key, value in state.items()
            if str(key).startswith("module.")
        }
        if not state:
            raise ValueError("TFCU checkpoint has no module-prefixed model weights")
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError(
                "TFCU checkpoint/model mismatch: "
                f"missing={loaded.missing_keys[:8]}, "
                f"unexpected={loaded.unexpected_keys[:8]}"
            )

    def predict_video(self, video_path):
        work_dir = Path(self.work_dir or "/tmp/fakeparts-work") / Path(video_path).stem
        work_dir.mkdir(parents=True, exist_ok=True)
        return self.predict_video_with_workdir(video_path, work_dir)

    def predict_video_with_workdir(self, video_path, work_dir):
        import cv2
        import numpy as np

        video_path = Path(video_path)
        work_dir = Path(work_dir)
        aligned_dir = work_dir / "aligned"
        landmark_dir = work_dir / "landmarks"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        landmark_dir.mkdir(parents=True, exist_ok=True)
        frames = self._prepare_frames(video_path, aligned_dir, landmark_dir, cv2, np)
        clip_count = frames // self.frames_per_clip
        if clip_count == 0:
            raise RuntimeError(
                f"TFCU needs at least {self.frames_per_clip} decodable frames: {video_path}"
            )

        clips = []
        landmark_clips = []
        mean = self.torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for clip_index in range(clip_count):
            image_tensors = []
            point_tensors = []
            for offset in range(self.frames_per_clip):
                frame_index = clip_index * self.frames_per_clip + offset
                image = cv2.imread(str(aligned_dir / f"{frame_index:08d}.jpg"))
                if image is None:
                    raise RuntimeError(f"missing prepared TFCU face frame {frame_index}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.image_size, self.image_size))
                tensor = self.torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
                image_tensors.append((tensor - mean) / std)
                point_tensors.append(self.torch.from_numpy(
                    np.load(landmark_dir / f"{frame_index:08d}.npy").copy()
                ).float())
            clips.append(self.torch.stack(image_tensors))
            landmark_clips.append(self.torch.stack(point_tensors))

        video = self.torch.stack(clips).unsqueeze(0)
        landmarks = self.torch.stack(landmark_clips).unsqueeze(0).to(
            self.device_obj, non_blocking=True)
        with self.torch.inference_mode():
            while True:
                try:
                    logits = self.model(
                        video, landmark=landmarks, eval=True,
                        frame_batch_size=self.feature_batch_size,
                    )
                    break
                except Exception as error:
                    if not is_cuda_oom(error) or self.feature_batch_size <= 1:
                        raise
                    self.feature_batch_size = max(1, self.feature_batch_size // 2)
                    self.torch.cuda.empty_cache()
                    print(
                        f"[TFCU] CUDA OOM; reducing feature frame batch to "
                        f"{self.feature_batch_size}", flush=True,
                    )
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
            raise ValueError(f"TFCU expected [1, clips*4, 2] logits, got {tuple(logits.shape)}")
        frame_fake_probabilities = self.torch.softmax(logits, dim=2)[:, :, 1]
        return self.result(frame_fake_probabilities.mean().item())

    def _prepare_frames(self, video_path, aligned_dir, landmark_dir, cv2, np):
        frame_cache_dir = None
        if getattr(self, "frame_cache_root", None) is not None:
            frame_cache_dir = cache_video_frames(video_path, self.frame_cache_root)
            cached_frames = sorted(frame_cache_dir.glob("frame_*.jpg"))
            if not cached_frames:
                raise RuntimeError(f"no cached frames found for TFCU video: {video_path}")
            capture = None
        else:
            cached_frames = []
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {video_path}")
        prepared = 0
        last_aligned = None
        last_landmarks = self._five_points(self.default_landmarks, np)
        try:
            frame_index = 0
            while True:
                if frame_cache_dir is not None:
                    if frame_index >= len(cached_frames):
                        break
                    frame = cv2.imread(str(cached_frames[frame_index]))
                    ok = frame is not None
                else:
                    ok, frame = capture.read()
                if not ok:
                    break
                aligned, landmarks = self._prepare_face(frame, last_aligned, last_landmarks, cv2, np)
                if aligned is not None:
                    last_aligned = aligned
                    last_landmarks = landmarks
                if not cv2.imwrite(str(aligned_dir / f"{frame_index:08d}.jpg"), aligned,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to stage TFCU frame {frame_index}")
                np.save(landmark_dir / f"{frame_index:08d}.npy", landmarks)
                frame_index += 1
                prepared += 1
        finally:
            if capture is not None:
                capture.release()
        if prepared == 0:
            raise RuntimeError(f"video has no decodable frames: {video_path}")
        return prepared

    def _prepare_face(self, frame, fallback_image, fallback_landmarks, cv2, np):
        faces = self.face_detector.detect(frame)
        faces = [face for face in faces if float(face[2]) >= 0.5][:10]
        if not faces:
            if fallback_image is not None:
                return fallback_image, fallback_landmarks
            image = cv2.resize(frame, (self.image_size, self.image_size))
            return image, fallback_landmarks

        bbox, five_landmarks, _ = faces[0]
        aligned, _, transformed_bbox = self.face_alignment.crop_aligned(
            frame, np.asarray(five_landmarks), landmarks_68=None,
            bboxes=np.asarray(bbox), aligned_image_size=self.image_size,
        )
        x0, y0, x1, y1 = map(float, transformed_bbox)
        width, height = x1 - x0, y1 - y0
        x0 = max(0, int(x0 - width / 8))
        y0 = max(0, int(y0 - height / 8))
        x1 = min(aligned.shape[1], int(x1 + width / 8))
        y1 = min(aligned.shape[0], int(y1 + height / 8))
        crop = aligned[y0:y1, x0:x1]
        if crop.size == 0:
            if fallback_image is not None:
                return fallback_image, fallback_landmarks
            return cv2.resize(frame, (self.image_size, self.image_size)), fallback_landmarks
        crop = cv2.resize(crop, (self.image_size, self.image_size))

        rectangles = self.landmark_detector(crop, 0)
        if rectangles:
            rectangle = max(
                rectangles,
                key=lambda item: max(0, item.right() - item.left())
                * max(0, item.bottom() - item.top()),
            )
            shape = self.landmark_predictor(crop, rectangle)
            landmarks81 = np.asarray([
                [shape.part(index).x, shape.part(index).y] for index in range(81)
            ], dtype=np.float32)
            return crop, self._five_points(landmarks81, np)
        if fallback_image is not None:
            return crop, fallback_landmarks
        return crop, self._five_points(self.default_landmarks, np)

    @staticmethod
    def _five_points(landmarks81, np):
        groups = (range(36, 42), range(42, 48), (30,), (48,), (54,))
        return np.stack([np.asarray(landmarks81)[list(group)].mean(axis=0) for group in groups])
