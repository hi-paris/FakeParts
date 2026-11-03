from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence


def is_cuda_oom(error: Exception) -> bool:
    """Return whether an inference exception is a CUDA allocation failure."""
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


class Detector(ABC):
    """Common detector adapter; scores are fake probabilities in [0, 1]."""
    output_level = "frame"

    def __init__(self, ckpt_path: str | Path, device: str = "cuda",
                 threshold: float = 0.5, crop_size: int | None = None,
                 work_dir: str | Path | None = None,
                 ckpt_paths: Sequence[str | Path] | None = None,
                 version: str = "default") -> None:
        self.ckpt_path = Path(ckpt_path)
        self.ckpt_paths = tuple(map(Path, ckpt_paths or (ckpt_path,)))
        self.device = device
        self.threshold = threshold
        self.crop_size = crop_size
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.version = version

    def result(self, score: float) -> dict[str, float | int]:
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"fake score must be in [0, 1], got {score}")
        return {"score": score, "prediction": int(score >= self.threshold)}

    def predict_frames(self, img_list: Sequence[Any]) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} is not a frame detector")

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} is not a video detector")


class FrameDetector(Detector):
    output_level = "frame"

    @abstractmethod
    def predict_frames(self, img_list: Sequence[Any]) -> list[dict[str, Any]]:
        """Return one standard result for each RGB PIL image."""


class VideoDetector(Detector):
    output_level = "video"

    @abstractmethod
    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        """Return one standard result for a video."""

    def predict_videos(self, video_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
        """Default video batching; adapters may override with a true batched path."""
        return [self.predict_video(path) for path in video_paths]

    def predict_video_with_workdir(
        self, video_path: str | Path, work_dir: str | Path
    ) -> dict[str, Any]:
        """Run one video's prepare/inference stages in an isolated temporary directory.

        Multi-stage adapters can override this method and place decoded frames,
        reconstructed clips, or features under ``work_dir``. The runner removes
        that directory as soon as this call returns (or raises).
        """
        return self.predict_video(video_path)
