# Generation

This package contains the generation adapters and the shared runner. The runner dispatches to each method's `generate.py`; model dependencies and checkpoints remain method-specific.

## Methods

| Task | Methods |
| --- | --- |
| Face swap | HunyuanPortrait, LivePortrait, InsightFace |
| Text or image to video | Allegro, CogVideoX, HunyuanVideo, LTX-Video, Mochi, Open-Sora, Wan 2.1 |
| Extrapolation | Cosmos-Predict2 |
| Interpolation | Framer |
| Inpainting | DiffuEraser, ProPainter, ROVI |
| Outpainting | AkiRA |
| Video editing | AnyV2V, RAVE |

## Run

Run one sample with its method and required inputs:

```bash
bash scripts/generate.sh --model framer --input path/to/video.mp4
```

For batch runs, pass a JSONL sample manifest. Use `--dry-run` to inspect planned commands without starting model inference.

The shared setup script currently supports selected generator environments and weights:

```bash
bash scripts/set_up_generator.sh --help
bash scripts/set_up_generator.sh --methods insightface
```

For Slurm, submit the same scripts and pass method arguments after the script name:

```bash
sbatch scripts/generate.sh --model framer --input path/to/video.mp4
```

## Inputs

This document describes the input contract for each registered generation method. It is a schema guide, not a record of benchmark sample mappings. For sample-level mappings, provide a JSONL manifest to `scripts/generate.py`.

Each JSONL line represents one sample. Use paths to existing input files and a stable `sample_id`:

```json
{"sample_id":"sample-001","input":"/data/prompt.txt"}
```

The fields required by each method are listed below. `prompt` can be supplied directly or loaded from a text file when the method accepts prompts. Paths may be absolute or relative to the current working directory.

| Task | Method | Required inputs | Optional inputs |
| --- | --- | --- | --- |
| Face swap | HunyuanPortrait | `source_video`, `source_image` | — |
| Face swap | LivePortrait | `source_video`, `source_image` | — |
| Face swap | InsightFace | `source_video`, `source_image` | — |
| Text to video | Allegro, CogVideoX, HunyuanVideo, LTX-Video, Mochi | `prompt` | — |
| Text to video | Open-Sora, Wan 2.1 | `prompt` | `task` |
| Image to video | Open-Sora, Wan 2.1 | `input` (image) | `task`, `prompt` |
| Video editing | AnyV2V | `source_video`, `prompt`, and either `edited_first_frame` or `edit_prompt` | `strength`, `image_model`, `output_name`, `device` |
| Video extrapolation | Cosmos-Predict2 | `source_video`, `prompt` | — |
| Video interpolation | Framer | `source_video` | — |
| Video inpainting | DiffuEraser | `source_video`, `mask` | — |
| Video inpainting or outpainting | ProPainter | `source_video` | `mask`, `task` |
| Video editing | ROVI, RAVE | `source_video`, `prompt` | — |
| Image outpainting | AkiRA | `source_image`, `trajectory` | — |

The shared runner writes outputs under `FakePartsBench/generated/<method>/` by default. Use `--output-dir` to select another location. Each method's adapter defines its own model settings and checkpoint loading; the shared runner does not infer missing inputs from dataset annotations.
