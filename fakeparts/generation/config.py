"""Registry for the local generation entry points and their input contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fakeparts.naming import GENERATORS, TASKS


GENERATION_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner_id: str
    task: str
    entrypoint: Path
    kind: str
    input_extensions: tuple[str, ...]
    input_field: str | None = None
    source_field: str | None = None
    mask_field: str | None = None
    trajectory_field: str | None = None
    input_flag: str | None = None
    source_flag: str | None = None
    mask_flag: str | None = None
    trajectory_flag: str | None = None
    task_flag: str | None = None
    strength_flag: str | None = None
    output_name_flag: str | None = None
    edit_prompt_flag: str | None = None
    image_model_flag: str | None = None
    device_flag: str | None = None

    @property
    def full_name(self) -> str:
        return GENERATORS.get(self.name, (self.name, self.name))[0]

    @property
    def short_name(self) -> str:
        return GENERATORS.get(self.name, (self.name, self.name))[1]

    def task_names(self, task: str) -> tuple[str, str]:
        return TASKS.get(task, (task, task))


def _spec(
    name: str,
    task: str,
    relative_entrypoint: str,
    kind: str,
    extensions: tuple[str, ...],
    runner_id: str | None = None,
    **kwargs: str,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        runner_id=runner_id or name,
        task=task,
        entrypoint=GENERATION_ROOT / relative_entrypoint,
        kind=kind,
        input_extensions=extensions,
        **kwargs,
    )


# These contracts are intentionally limited to model entry points whose CLI is
# present in this checkout. The runner does not install model dependencies.
MODEL_SPECS: dict[str, ModelSpec] = {
    "hunyuanportrait": _spec(
        "hunyuanportrait", "faceswap", "faceswap/hunyuanportrait/generate.py", "paired",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video",
        source_field="source_image", input_flag="--driving", source_flag="--source",
    ),
    "liveportrait": _spec(
        "liveportrait", "faceswap", "faceswap/liveportrait/generate.py", "paired",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video",
        source_field="source_image", input_flag="--driving", source_flag="--source",
    ),
    "insightface": _spec(
        "insightface", "faceswap", "faceswap/insightface/generate.py", "paired",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video",
        source_field="source_image", input_flag="--input", source_flag="--source",
    ),
    "allegroai": _spec(
        "allegroai", "t2v", "t2v-it2v/allegroai/generate.py", "prompt", (".txt",), runner_id="allegroai",
    ),
    "cogvidx": _spec(
        "cogvidx", "t2v", "t2v-it2v/cogvidx/generate.py", "prompt", (".txt",), runner_id="cogvidx",
    ),
    "hunyuanvid": _spec(
        "hunyuanvid", "t2v", "t2v-it2v/hunyuanvid/generate.py", "prompt", (".txt",), runner_id="hunyuanvid",
    ),
    "ltx_video": _spec(
        "ltx_video", "t2v", "t2v-it2v/ltx_video/generate.py", "prompt", (".txt",),
    ),
    "mochi1": _spec(
        "mochi1", "t2v", "t2v-it2v/mochi1/generate.py", "prompt", (".txt",), runner_id="mochi1",
    ),
    "opensora": _spec(
        "opensora", "t2v/it2v", "t2v-it2v/opensora/generate.py", "prompt_or_image",
        (".txt", ".png", ".jpg", ".jpeg", ".webp"), input_flag="--input", task_flag="--task",
    ),
    "wan21": _spec(
        "wan21", "t2v/it2v", "t2v-it2v/wan21/generate.py", "prompt_or_image",
        (".txt", ".png", ".jpg", ".jpeg", ".webp"), input_flag="--input", task_flag="--task",
    ),
    "anyv2v": _spec(
        "anyv2v", "style", "style/anyv2v/generate.py", "conditioned_input_prompt",
        (".mp4",), input_field="source_video", source_field="edited_first_frame",
        input_flag="--input", source_flag="--edited-first-frame", strength_flag="--strength",
        output_name_flag="--output-name", edit_prompt_flag="--edit-prompt",
        image_model_flag="--image-model", device_flag="--device",
    ),
    "cosmos2": _spec(
        "cosmos2", "extrapolation", "extrapolation/cosmos2/generate.py",
        "input_prompt", (".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"),
        input_field="source_video", input_flag="--input", runner_id="cosmos2",
    ),
    "framer": _spec(
        "framer", "interpolation", "interpolation/framer/generate.py", "video_endpoints",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video", input_flag="--input",
    ),
    "diffueraser": _spec(
        "diffueraser", "inpainting", "inpainting/diffueraser/generate.py", "masked_input",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video",
        mask_field="mask", input_flag="--input", mask_flag="--mask",
    ),
    "propainter": _spec(
        "propainter", "inpainting", "inpainting/propainter/generate.py", "masked_input",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video",
        mask_field="mask", input_flag="--input", mask_flag="--mask", task_flag="--task",
    ),
    "rovi": _spec(
        "rovi", "inpainting", "inpainting/rovi/generate.py", "input_prompt",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video", input_flag="--input",
    ),
    "akira": _spec(
        "akira", "outpainting", "outpainting/akira/generate.py", "trajectory_input",
        (".jpg", ".jpeg", ".png", ".webp"), input_field="source_image",
        trajectory_field="trajectory", input_flag="--input", trajectory_flag="--trajectory",
    ),
    "rave": _spec(
        "rave", "style", "style/rave/generate.py", "input_prompt",
        (".mp4", ".mov", ".avi", ".mkv"), input_field="source_video", input_flag="--input",
    ),
}


def get_model(name: str) -> ModelSpec:
    key = name.strip().lower().replace("-", "_")
    aliases = {"wan_2_1": "wan21", "allegro": "allegroai", "cogvideox": "cogvidx", "hunyuanvideo": "hunyuanvid", "mochi": "mochi1", "cosmos_predict2": "cosmos2"}
    key = aliases.get(key, key)
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {name!r}. Supported models: {choices}") from exc
