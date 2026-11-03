#!/usr/bin/env python3
"""Generate videos through the checked-in, model-specific adapter scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fakeparts.generation.adapters import IMAGE_EXTENSIONS, GenerationInput, ModelAdapter, prompt_from_path
from fakeparts.generation.base import GenerationJob, Generator
from fakeparts.generation.config import MODEL_SPECS, get_model
from fakeparts.naming import canonical_generator_id

TASK_ORDER = ("faceswap", "t2v", "it2v", "extrapolation", "interpolation", "inpainting", "outpainting", "style")


def input_contract(spec: Any, task: str) -> str:
    if spec.kind == "paired":
        return "video + source image"
    if spec.kind == "prompt":
        return "prompt"
    if spec.kind == "prompt_or_image":
        return "image + prompt" if task == "it2v" else "prompt"
    if spec.kind == "conditioned_input_prompt":
        return "video + prompt + edited image or edit prompt"
    if spec.kind == "input_prompt":
        media = "video/image" if any(ext in IMAGE_EXTENSIONS for ext in spec.input_extensions) else "video"
        return f"{media} + prompt"
    if spec.kind == "video_endpoints":
        return "video"
    if spec.kind == "masked_input":
        return "video + mask (ProPainter outpainting: video)"
    if spec.kind == "trajectory_input":
        return "image + trajectory"
    return spec.kind


def model_catalog_rows() -> list[tuple[str, str, str, str, str]]:
    rows = []
    for model_name, spec in sorted(MODEL_SPECS.items()):
        tasks = spec.task.split("/")
        for task in tasks:
            model_id = f"{model_name}:{task}" if len(tasks) > 1 else model_name
            rows.append((model_id, task, model_name, input_contract(spec, task), spec.kind))
    return rows


def model_catalog_help() -> str:
    groups: dict[str, list[str]] = {}
    for model_id, task, _model, required_input, _kind in model_catalog_rows():
        groups.setdefault(task, []).append(f"    {model_id}: {required_input}")
    return "Registered model IDs by task:\n" + "\n".join(
        f"  {task}:\n" + "\n".join(groups[task])
        for task in (*TASK_ORDER, *sorted(set(groups) - set(TASK_ORDER))) if task in groups
    )


def parse_model_id(model_id: str) -> tuple[Any, str | None]:
    model_name, separator, requested_task = model_id.partition(":")
    spec = get_model(model_name)
    if not separator:
        return spec, None
    if requested_task not in spec.task.split("/"):
        raise ValueError(
            f"Unknown task-specific model ID {model_id!r}; expected one of: "
            f"{', '.join(f'{spec.name}:{task}' for task in spec.task.split('/'))}"
        )
    return spec, requested_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=model_catalog_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    method = parser.add_argument_group("Method selection")
    method.add_argument("--model", default=os.getenv("FAKEPARTS_MODEL"),
                        help="One model ID from the task list below (or FAKEPARTS_MODEL).")

    inputs = parser.add_argument_group("Inputs (required combination depends on the selected model)")
    inputs.add_argument("--input", type=Path, default=os.getenv("FAKEPARTS_INPUT"),
                        help="One input file or a folder; folders are searched recursively.")
    inputs.add_argument("--source", type=Path, default=os.getenv("FAKEPARTS_SOURCE"),
                        help="Source image for paired methods or edited first frame for AnyV2V.")
    inputs.add_argument("--mask", type=Path, default=os.getenv("FAKEPARTS_MASK"),
                        help="Mask file for inpainting methods that require one.")
    inputs.add_argument("--trajectory", type=Path, default=os.getenv("FAKEPARTS_TRAJECTORY"),
                        help="Camera trajectory file for AkiRA.")
    inputs.add_argument("--prompt", default=os.getenv("FAKEPARTS_PROMPT"), help="Prompt text.")
    inputs.add_argument("--prompt-file", type=Path, default=os.getenv("FAKEPARTS_PROMPT_FILE"),
                        help="UTF-8 text file containing a prompt.")

    batch = parser.add_argument_group("Manifest and batch selection")
    batch.add_argument("--manifest", type=Path, default=os.getenv("FAKEPARTS_MANIFEST"),
                       help="JSONL manifest containing sample-level input mappings.")
    batch.add_argument("--sample-id", action="append", default=[],
                       help="Select a manifest sample; may be repeated.")
    batch.add_argument("--prompt-id", default=os.getenv("FAKEPARTS_PROMPT_ID"),
                       help="Output ID for a single direct input; otherwise derived from its filename.")

    task = parser.add_argument_group("Task-specific options")
    task.add_argument("--task", default=os.getenv("FAKEPARTS_TASK"),
                      help="Task override, e.g. ProPainter inpainting or outpainting.")
    task.add_argument("--strength", default=os.getenv("FAKEPARTS_STRENGTH"),
                      choices=("0.3", "0.7", "1.0"), help="AnyV2V editing strength.")
    task.add_argument("--output-name", default=os.getenv("FAKEPARTS_OUTPUT_NAME"),
                      help="AnyV2V output filename override.")
    task.add_argument("--edit-prompt", default=os.getenv("FAKEPARTS_EDIT_PROMPT"),
                      help="AnyV2V image-edit prompt when no edited first frame is supplied.")
    task.add_argument("--image-model", default=os.getenv("FAKEPARTS_IMAGE_MODEL"),
                      choices=("instructpix2pix", "magicbrush", "cosxl"),
                      help="AnyV2V first-frame image editor.")

    output = parser.add_argument_group("Output and run control")
    output.add_argument("--output-dir", type=Path, default=os.getenv("FAKEPARTS_OUTPUT_DIR"),
                         help="Output directory (default: FakePartsBench/generated/<model>).")
    output.add_argument("--seed", type=int, default=int(os.getenv("FAKEPARTS_SEED", "0")),
                        help="Random seed (default: 0).")
    output.add_argument("--device", default=os.getenv("FAKEPARTS_DEVICE"),
                        help="Device override for adapters that expose one.")
    output.add_argument("--limit", type=int, help="Run at most N runnable samples.")
    output.add_argument("--dry-run", action="store_true", help="Print planned commands without loading models.")
    output.add_argument("--continue-on-error", action="store_true", help="Continue after a sample fails.")
    output.add_argument("--skip-existing", action="store_true",
                        help="Skip non-empty output MP4s already in --output-dir.")
    output.add_argument("--strict", action="store_true",
                        help="Fail if selected manifest rows lack required inputs.")

    parser.add_argument("--list-models", action="store_true",
                        help="Print the model/task/input/class table.")
    return parser


def print_model_catalog() -> None:
    rows = model_catalog_rows()
    headers = ("ID", "method", "full name", "short name", "input", "class")
    display_rows = []
    for model_id, task, model_name, required_input, kind in rows:
        spec = MODEL_SPECS[model_name]
        task_full, task_short = spec.task_names(task)
        display_rows.append((model_id, task_short, spec.full_name, spec.short_name, required_input, kind))
    widths = [max(len(headers[i]), *(len(row[i]) for row in display_rows)) for i in range(len(headers))]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in display_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def local_path(value: str | Path, *, dataset_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if dataset_root is not None:
        candidate = dataset_root / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {resolved}:{line_number}")
            rows.append(row)
    return rows


def row_input(row: dict[str, Any], field: str | None, dataset_root: Path) -> Path | None:
    value = row.get(field) if field else None
    if not value:
        return None
    path = local_path(value, dataset_root=dataset_root)
    return path if path.is_file() else None


def manifest_inputs(
    rows: Iterable[dict[str, Any]],
    *,
    spec: Any,
    dataset_root: Path,
    explicit_prompt: str | None,
    explicit_strength: str | None,
    explicit_task: str | None,
    selected_ids: set[str],
) -> tuple[list[GenerationInput], list[str]]:
    inputs: list[GenerationInput] = []
    skipped: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("experiment_id") or row.get("file_name") or "")
        if selected_ids and sample_id not in selected_ids and sample_id.rsplit(":", 1)[-1] not in selected_ids:
            continue
        if row.get("method"):
            row_method = canonical_generator_id(str(row["method"]))
            allowed_methods = {spec.name}
            if spec.name == "anyv2v":
                allowed_methods.add("anyv2v_style_transfer")
            if row_method not in allowed_methods:
                skipped.append(f"{sample_id}: method does not match --model {spec.name}")
                continue

        input_path = row_input(row, spec.input_field, dataset_root)
        if input_path is None and spec.name == "anyv2v":
            input_path = row_input(row, "video_path", dataset_root)
        source_path = row_input(row, spec.source_field, dataset_root)
        mask_path = row_input(row, spec.mask_field, dataset_root)
        trajectory_path = row_input(row, spec.trajectory_field, dataset_root)
        if source_path is None and spec.name == "anyv2v":
            source_path = row_input(row, "edited_first_frame_path", dataset_root)
        prompt: str | None = explicit_prompt
        prompt_value = row.get("prompt")
        if prompt_value is None and spec.name == "anyv2v":
            prompt_value = row.get("editing_prompt") or row.get("anyv2v_prompt")
        if prompt is None and prompt_value:
            prompt_path = local_path(prompt_value, dataset_root=dataset_root)
            if prompt_path.is_file():
                try:
                    prompt = prompt_from_path(prompt_path)
                except ValueError as exc:
                    skipped.append(f"{sample_id}: {exc}")
                    continue
            elif isinstance(prompt_value, str) and (
                prompt_path.suffix.lower() != ".txt"
                and "/" not in prompt_value
                and "\\" not in prompt_value
            ):
                # Some manifests store prompt text directly; most store a path.
                prompt = prompt_value
            else:
                skipped.append(f"{sample_id}: prompt path is missing: {prompt_path}")
                continue

        if spec.kind == "prompt_or_image" and row.get("source_image"):
            input_path = row_input(row, "source_image", dataset_root)
        strength_value = explicit_strength
        if strength_value is None:
            row_strength = row.get("strength")
            if row_strength is None and spec.name == "anyv2v":
                row_strength = row.get("style_strength")
            if row_strength is not None:
                strength_value = str(row_strength)
        edit_prompt_value = (
            row.get("edit_prompt") or row.get("edit_image_prompt") or row.get("style_instruction")
            if spec.name == "anyv2v" else None
        )
        item = GenerationInput(
            sample_id=sample_id,
            input_path=input_path,
            source_path=source_path,
            mask_path=mask_path,
            trajectory_path=trajectory_path,
            prompt=prompt,
            task=str(explicit_task or row.get("task") or "t2v"),
            strength=strength_value,
            output_name=str(row["output_name"]) if row.get("output_name") else None,
            edit_prompt=str(edit_prompt_value) if edit_prompt_value else None,
            image_model=str(row["image_model"]) if row.get("image_model") else None,
            seed=int(row["seed"]) if row.get("seed") is not None else None,
            device=str(row["device"]) if row.get("device") else None,
        )
        inputs.append(item)
    return inputs, skipped


def direct_inputs(args: argparse.Namespace, spec: Any, prompt: str | None) -> list[GenerationInput]:
    input_path = local_path(args.input) if args.input is not None else None
    source_path = local_path(args.source) if args.source is not None else None
    mask_path = local_path(args.mask) if args.mask is not None else None
    trajectory_path = local_path(args.trajectory) if args.trajectory is not None else None
    if input_path is not None and not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if source_path is not None and not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")
    if mask_path is not None and not mask_path.exists():
        raise ValueError(f"Mask path does not exist: {mask_path}")
    if trajectory_path is not None and not trajectory_path.exists():
        raise ValueError(f"Trajectory path does not exist: {trajectory_path}")

    candidates: list[Path]
    if input_path is None:
        candidates = []
    elif input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in spec.input_extensions
        )
    if input_path is not None and input_path.is_dir() and not candidates:
        raise ValueError(f"No supported inputs found under {input_path}")

    if spec.kind == "paired" and source_path is None:
        raise ValueError(f"--source is required for model {spec.name}")
    if spec.kind == "masked_input" and mask_path is None and not (spec.name == "propainter" and args.task == "outpainting"):
        raise ValueError(f"--mask is required for model {spec.name}")
    if spec.kind == "trajectory_input" and trajectory_path is None:
        raise ValueError(f"--trajectory is required for model {spec.name}")
    if spec.kind in {"prompt", "prompt_or_image"} and prompt is None and not candidates:
        raise ValueError("Provide --prompt, --prompt-file or a prompt .txt input")
    if spec.kind == "input_prompt" and (input_path is None or prompt is None):
        raise ValueError(f"Model {spec.name} requires both --input and --prompt")
    if not candidates:
        sample_id = args.prompt_id or "sample"
        return [GenerationInput(sample_id=sample_id, prompt=prompt)]

    items: list[GenerationInput] = []
    for path in candidates:
        sample_id = args.prompt_id if args.prompt_id and len(candidates) == 1 else path.stem
        item_source = source_path
        if spec.kind in {"paired", "conditioned_input_prompt"} and source_path is not None and source_path.is_dir():
            matches = [p for p in source_path.rglob("*") if p.is_file() and p.stem == path.stem]
            if len(matches) != 1:
                raise ValueError(f"Expected one source image matching {path.stem!r} under {source_path}; found {len(matches)}")
            item_source = matches[0]

        item_prompt = prompt
        if path.suffix.lower() == ".txt":
            item_prompt = prompt_from_path(path)
            actual_input = None
        else:
            actual_input = path
        items.append(GenerationInput(
            sample_id=sample_id,
            input_path=actual_input,
            source_path=item_source,
            mask_path=mask_path,
            trajectory_path=trajectory_path,
            prompt=item_prompt,
            strength=args.strength,
            output_name=args.output_name,
            edit_prompt=args.edit_prompt,
            image_model=args.image_model,
            device=args.device,
            task=args.task,
        ))
    return items


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        print_model_catalog()
        return 0
    if not args.model:
        parser.error("--model or FAKEPARTS_MODEL is required")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        spec, model_task = parse_model_id(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    if model_task is not None:
        if args.task and args.task != model_task:
            parser.error(f"--model {args.model} conflicts with --task {args.task}")
        args.task = model_task

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt_file = local_path(args.prompt_file)
        if not prompt_file.is_file():
            parser.error(f"Prompt file does not exist: {prompt_file}")
        if prompt is not None:
            parser.error("Use either --prompt or --prompt-file, not both")
        prompt = prompt_from_path(prompt_file)

    output_dir = local_path(args.output_dir) if args.output_dir else REPO_ROOT / "FakePartsBench/generated" / spec.name
    adapter = ModelAdapter(
        spec,
        python=os.getenv("FAKEPARTS_PYTHON", sys.executable),
        output_dir=output_dir,
    )

    skipped: list[str] = []
    if args.manifest is not None:
        if any(value is not None for value in (args.input, args.source, args.mask, args.trajectory)):
            parser.error("Use --manifest with its mapped inputs, or use direct --input/--source/--mask/--trajectory mode")
        manifest_path = local_path(args.manifest)
        if not manifest_path.is_file():
            parser.error(f"Manifest does not exist: {manifest_path}")
        dataset_root = REPO_ROOT / "FakePartsBench"
        items, skipped = manifest_inputs(
            load_manifest(manifest_path),
            spec=spec,
            dataset_root=dataset_root,
            explicit_prompt=prompt,
            explicit_strength=args.strength,
            explicit_task=args.task,
            selected_ids=set(args.sample_id),
        )
    else:
        items = direct_inputs(args, spec, prompt)

    jobs: list[GenerationJob] = []
    for item in items:
        if spec.name in {"opensora", "wan21"} and args.task in {"t2v", "it2v"}:
            has_image = item.input_path is not None and item.input_path.suffix.lower() in IMAGE_EXTENSIONS
            input_task = "it2v" if has_image else "t2v"
            if input_task != args.task:
                skipped.append(
                    f"{item.sample_id}: --model {spec.name}:{args.task} requires "
                    f"{('an image' if args.task == 'it2v' else 'prompt-only input')}"
                )
                continue
        try:
            jobs.append(adapter.build_job(item, seed=args.seed))
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(f"{item.sample_id}: {exc}")
    for message in skipped:
        print(f"[skip] {message}", file=sys.stderr)
    if args.strict and skipped:
        print(f"[generate] strict mode: {len(skipped)} selected input(s) are incomplete", file=sys.stderr)
        return 2
    if args.sample_id and args.manifest is not None and not items:
        parser.error("No manifest rows match --sample-id")
    if args.limit is not None:
        jobs = jobs[:args.limit]
    if not jobs:
        print("[generate] no runnable inputs", file=sys.stderr)
        return 2
    output_names = []
    for job in jobs:
        command = job.command
        if "--output-name" in command:
            output_names.append(command[command.index("--output-name") + 1])
        else:
            prompt_id = command[command.index("--prompt-id") + 1]
            seed = command[command.index("--seed") + 1]
            output_names.append(f"{prompt_id}_{seed}.mp4")
    if len(output_names) != len(set(output_names)):
        print("[generate] duplicate output filenames in selected inputs; refusing ambiguous batch", file=sys.stderr)
        return 2

    if args.skip_existing:
        pending_jobs: list[GenerationJob] = []
        skipped_existing = 0
        for job, output_name in zip(jobs, output_names, strict=True):
            output_path = output_dir / output_name
            if output_path.is_file() and output_path.stat().st_size > 0:
                print(f"[skip] output already exists: {output_path}", file=sys.stderr)
                skipped_existing += 1
            else:
                pending_jobs.append(job)
        jobs = pending_jobs
        if not jobs and skipped_existing:
            print(f"[generate] all {skipped_existing} selected outputs already exist", flush=True)
            return 0

    runner = Generator(
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        setup_method=spec.runner_id,
        setup_available=spec.runner_id in {"hunyuanportrait", "liveportrait"},
    )
    return runner.run(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
