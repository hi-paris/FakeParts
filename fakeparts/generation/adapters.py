"""Translate benchmark inputs into the existing per-model CLI contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .base import GenerationJob, command_for
from .config import ModelSpec


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class GenerationInput:
    sample_id: str
    input_path: Path | None = None
    source_path: Path | None = None
    mask_path: Path | None = None
    trajectory_path: Path | None = None
    prompt: str | None = None
    task: str | None = None
    strength: str | None = None
    output_name: str | None = None
    edit_prompt: str | None = None
    image_model: str | None = None
    seed: int | None = None
    device: str | None = None


def prompt_from_path(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def prompt_id_for(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_-")
    if not safe:
        raise ValueError(f"Cannot derive a prompt ID from {sample_id!r}")
    if not safe[0].isalnum():
        safe = "sample_" + safe
    return safe[:120]


class ModelAdapter:
    """Build one subprocess call for a model-specific checked-in adapter."""

    def __init__(self, spec: ModelSpec, *, python: str, output_dir: Path) -> None:
        self.spec = spec
        self.python = python
        self.output_dir = output_dir

    def build_job(self, item: GenerationInput, *, seed: int) -> GenerationJob:
        if not self.spec.entrypoint.is_file():
            raise FileNotFoundError(f"Model adapter is missing: {self.spec.entrypoint}")
        if item.output_name is not None and (
            Path(item.output_name).name != item.output_name
            or not item.output_name.lower().endswith(".mp4")
        ):
            raise ValueError(f"Expected output_name to be a plain .mp4 filename, got {item.output_name!r}")

        prompt = item.prompt
        input_path = item.input_path
        if input_path is not None and input_path.suffix.lower() == ".txt":
            if prompt is not None:
                raise ValueError(f"Both --prompt and prompt file were supplied for {item.sample_id}")
            prompt = prompt_from_path(input_path)
            input_path = None

        argv: list[str] = []
        used_inputs: list[str] = []
        task = item.task

        if self.spec.kind == "paired":
            if input_path is None or item.source_path is None:
                raise ValueError("Face-swap generation needs a driving video and a source image")
            if input_path.suffix.lower() not in self.spec.input_extensions:
                raise ValueError(f"Expected a driving video, got {input_path}")
            if item.source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Expected a source image, got {item.source_path}")
            argv.extend([self.spec.source_flag or "--source", str(item.source_path)])
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            used_inputs.extend((str(item.source_path), str(input_path)))
        elif self.spec.kind == "prompt":
            if input_path is not None:
                raise ValueError(f"Prompt models accept .txt inputs, not {input_path}")
            if prompt is None:
                raise ValueError("This model needs --prompt or a .txt input")
            argv.extend(["--prompt", prompt])
        elif self.spec.kind == "prompt_or_image":
            if prompt is None:
                raise ValueError("This model needs --prompt or a .txt input")
            if input_path is not None:
                if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"Expected an image condition, got {input_path}")
                task = "it2v"
                argv.extend([self.spec.input_flag or "--input", str(input_path)])
                used_inputs.append(str(input_path))
            else:
                task = "t2v"
            argv.extend(["--prompt", prompt])
        elif self.spec.kind == "input_prompt":
            if input_path is None or prompt is None:
                raise ValueError("This model needs an image/video --input and --prompt")
            if input_path.suffix.lower() not in self.spec.input_extensions:
                raise ValueError(f"Unsupported input type for {self.spec.name}: {input_path}")
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            used_inputs.append(str(input_path))
        elif self.spec.kind == "video_endpoints":
            if input_path is None or input_path.suffix.lower() not in self.spec.input_extensions:
                raise ValueError("This model needs a supported video --input")
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            used_inputs.append(str(input_path))
        elif self.spec.kind == "masked_input":
            needs_mask = not (self.spec.name == "propainter" and task == "outpainting")
            if input_path is None or (needs_mask and item.mask_path is None):
                raise ValueError("This model needs both a video --input and file --mask")
            if input_path.suffix.lower() not in self.spec.input_extensions:
                raise ValueError(f"Unsupported video input for {self.spec.name}: {input_path}")
            if item.mask_path is not None and not item.mask_path.is_file():
                raise ValueError(f"Mask must be a file supported by {self.spec.name}: {item.mask_path}")
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            if item.mask_path is not None:
                argv.extend([self.spec.mask_flag or "--mask", str(item.mask_path)])
                used_inputs.append(str(item.mask_path))
            used_inputs.append(str(input_path))
        elif self.spec.kind == "trajectory_input":
            if input_path is None or item.trajectory_path is None:
                raise ValueError("This model needs both an image --input and --trajectory")
            if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Expected an image condition, got {input_path}")
            if not item.trajectory_path.is_file():
                raise ValueError(f"Trajectory must be a file: {item.trajectory_path}")
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            argv.extend([self.spec.trajectory_flag or "--trajectory", str(item.trajectory_path)])
            used_inputs.extend((str(input_path), str(item.trajectory_path)))
        elif self.spec.kind == "conditioned_input_prompt":
            if input_path is None or prompt is None or (item.source_path is None and item.edit_prompt is None):
                raise ValueError("This model needs a video, target prompt, and edited frame or image-edit prompt")
            if input_path.suffix.lower() not in self.spec.input_extensions:
                raise ValueError(f"Unsupported video input for {self.spec.name}: {input_path}")
            if item.source_path is not None and item.source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Expected an edited first-frame image, got {item.source_path}")
            argv.extend([self.spec.input_flag or "--input", str(input_path)])
            if item.source_path is not None:
                argv.extend([self.spec.source_flag or "--edited-first-frame", str(item.source_path)])
                used_inputs.append(str(item.source_path))
            if item.edit_prompt is not None:
                argv.extend([self.spec.edit_prompt_flag or "--edit-prompt", item.edit_prompt])
            if item.image_model is not None and self.spec.image_model_flag:
                argv.extend([self.spec.image_model_flag, item.image_model])
            argv.extend(["--prompt", prompt])
            used_inputs.append(str(input_path))
        if self.spec.kind == "input_prompt":
            if prompt is None:
                raise ValueError(f"This model needs a --prompt for {self.spec.name}")
            argv.extend(["--prompt", prompt])
        elif self.spec.kind == "masked_input" and prompt is not None:
            # DiffuEraser and ProPainter ignore text prompts; their CLI has no prompt flag.
            if self.spec.name not in {"diffueraser", "propainter"}:
                argv.extend(["--prompt", prompt])

        if self.spec.task_flag:
            if task is None:
                task = "inpainting" if self.spec.name == "propainter" else "t2v"
            argv.extend([self.spec.task_flag, task])
        if self.spec.strength_flag and item.strength is not None:
            argv.extend([self.spec.strength_flag, str(item.strength)])
        if self.spec.output_name_flag and item.output_name is not None:
            argv.extend([self.spec.output_name_flag, item.output_name])
        if self.spec.device_flag and item.device is not None:
            argv.extend([self.spec.device_flag, item.device])
        prompt_id = prompt_id_for(item.sample_id)
        seed = item.seed if item.seed is not None else seed
        argv.extend([
            "--prompt-id", prompt_id,
            "--seed", str(seed),
            "--output-dir", str(self.output_dir),
        ])
        entrypoint = str(self.spec.entrypoint)
        return GenerationJob(
            sample_id=item.sample_id,
            command=command_for(entrypoint, self.python, argv),
            inputs=tuple(used_inputs),
        )
