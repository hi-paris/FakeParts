from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from ...base import VideoDetector, is_cuda_oom
from ...utils import cache_video_frames


class _RaftArgs(SimpleNamespace):
    def __contains__(self, key):
        return isinstance(key, str) and hasattr(self, key)


class AIGVDet(VideoDetector):
    """AIGVDet RGB/RAFT-flow ensemble, aggregated over decoded video frames."""

    frame_batch_size = 8

    def set_inference_batch_size(self, batch_size: int) -> None:
        self.frame_batch_size = max(1, int(batch_size))

    def _predict_rgb_scores(self, batch):
        try:
            return self.rgb_model(batch).sigmoid().flatten()
        except Exception as error:
            if not is_cuda_oom(error) or batch.shape[0] <= 1:
                raise
            self.frame_batch_size = min(self.frame_batch_size, batch.shape[0] // 2)
            self.torch.cuda.empty_cache()
            print(
                f"[AIGVDet] CUDA OOM; reducing RGB frame batch to "
                f"{self.frame_batch_size}", flush=True,
            )
            split = self.frame_batch_size
            return self.torch.cat((
                self._predict_rgb_scores(batch[:split]),
                self._predict_rgb_scores(batch[split:]),
            ))

    def __init__(self, ckpt_path, device="cuda", threshold=0.5, crop_size=None,
                 work_dir=None, ckpt_paths=None, version="default"):
        super().__init__(ckpt_path, device, threshold, crop_size, work_dir, ckpt_paths, version)
        import torch

        vendor_root = Path(__file__).resolve().parent
        project_root = Path(__file__).resolve().parents[4]
        for dependency_root in (
            project_root / ".deps" / "detection-extra",
            project_root / ".deps" / "detection",
        ):
            if dependency_root.is_dir() and str(dependency_root) not in sys.path:
                sys.path.append(str(dependency_root))
        sys.path[:0] = [str(vendor_root), str(vendor_root / "core")]
        try:
            from core.raft import RAFT
            from core.utils1.utils import get_network
            from core.utils.utils import InputPadder
            from core.utils import flow_viz
        except Exception:
            sys.path.remove(str(vendor_root))
            sys.path.remove(str(vendor_root / "core"))
            raise

        if len(self.ckpt_paths) != 3:
            raise ValueError("AIGVDet requires RAFT, RGB-classifier, and flow-classifier checkpoints")
        self.torch = torch
        self.InputPadder = InputPadder
        self.flow_viz = flow_viz
        self.rgb_model = get_network("resnet50")
        self.flow_classifier = get_network("resnet50")
        self.rgb_model.load_state_dict(self._state(self.ckpt_paths[1]), strict=True)
        self.flow_classifier.load_state_dict(self._state(self.ckpt_paths[2]), strict=True)
        self.rgb_model.to(device).eval()
        self.flow_classifier.to(device).eval()

        self.flow_model = RAFT(_RaftArgs(
            small=False, mixed_precision=False, alternate_corr=False
        ))
        flow_state = self._state(self.ckpt_paths[0])
        self.flow_model.load_state_dict(flow_state, strict=False)
        self.flow_model.to(device).eval()
        self.device_obj = torch.device(device)

    @staticmethod
    def _state(path):
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        return state

    def predict_video(self, video_path):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import functional as TF

        frames = []
        if getattr(self, "frame_cache_root", None) is not None:
            cached_frames = cache_video_frames(video_path, self.frame_cache_root)
            frame_paths = sorted(cached_frames.glob("frame_*.jpg"))
            for frame_path in frame_paths:
                with Image.open(frame_path) as image:
                    image = image.convert("RGB")
                    frames.append(TF.to_tensor(
                        transforms.CenterCrop((448, 448))(image)))
        else:
            capture = cv2.VideoCapture(str(video_path))
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(TF.to_tensor(
                        transforms.CenterCrop((448, 448))(image)))
            finally:
                capture.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        raw = torch.stack(frames)
        normalized = TF.normalize(
            raw, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        rgb_score_sum = 0.0
        flow_score_sum = 0.0
        with torch.inference_mode():
            start = 0
            while start < len(normalized):
                stop = min(start + self.frame_batch_size, len(normalized))
                rgb = normalized[start:stop].to(self.device_obj)
                rgb_scores = self._predict_rgb_scores(rgb)
                rgb_score_sum += float(rgb_scores.sum().cpu())

                # RAFT only needs adjacent pairs. Keep decoded frames on CPU and
                # move one pair at a time to the GPU to bound peak memory for
                # long source videos.
                flow_frames = []
                for index in range(start, stop):
                    if len(normalized) == 1:
                        first = second = normalized[index:index + 1]
                    elif index < len(normalized) - 1:
                        first = normalized[index:index + 1]
                        second = normalized[index + 1:index + 2]
                    else:
                        first = normalized[index - 1:index]
                        second = normalized[index:index + 1]
                    first, second = first.to(self.device_obj), second.to(self.device_obj)
                    padder = self.InputPadder(first.shape)
                    first, second = padder.pad(first.float(), second.float())
                    _, flow_up = self.flow_model(first, second, iters=20, test_mode=True)
                    flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
                    rendered = self.flow_viz.flow_to_image(flow)
                    tensor = torch.from_numpy(rendered).permute(2, 0, 1).float() / 255.0
                    flow_frames.append(tensor * 2.0 - 1.0)
                optical = TF.normalize(
                    torch.stack(flow_frames), [0.5, 0.5, 0.5], [0.226, 0.226, 0.226]
                ).to(self.device_obj)
                flow_scores = self.flow_classifier(optical).sigmoid().flatten()
                flow_score_sum += float(flow_scores.sum().cpu())
                del rgb, rgb_scores, flow_frames, optical, flow_scores
                start = stop
            score = (rgb_score_sum + flow_score_sum) / (2 * len(normalized))
        return self.result(score)
