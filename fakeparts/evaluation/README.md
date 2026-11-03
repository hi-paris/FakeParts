# Video evaluation

These scripts evaluate video samples listed in JSONL or detector result CSVs.
They leave source videos in place and write metrics under `results/eval/` by
default. Evaluation-specific options and dependencies are documented here; per-sample dataset metadata remains in
`FakePartsBench/metadata.jsonl` and `FakePartsBench/manifests/`.

## Unified shell entry points

Run evaluation scripts directly with `bash`, or submit the same scripts with
`sbatch` from the repository root. Pass Slurm resource options before the script
path and evaluation options after it.

Prepare the shared evaluation environment once with:

```bash
bash scripts/evaluate/set_up.sh
sbatch scripts/evaluate/set_up.sh
```

```bash
bash scripts/evaluate/vbench.sh \
  --input-jsonl /path/to/manifest.jsonl

sbatch scripts/evaluate/fvd.sh \
  --input-jsonl /path/to/manifest.jsonl

bash scripts/evaluate/auc.sh \
  --method cnndet
```

VBench and FVD select the `label: fake` rows from the input JSONL. The AUC
script reads the selected detector folder `results/<family>/<detector>/`.
Fake files are named `<task>_<method>.<version>.csv`. The AUC config selects
the real source classes `tenk_youtube` and `youtube_vos_2019`, then scans
version-matched score CSVs and selects rows by `label=real` and the `real/`
source class in their metadata. It reuses existing scores regardless of
whether an older CSV is named `real_tenk_youtube.csv` or
`tenk_youtube.csv`; it does not run detection on real videos. It averages frame
scores per video, then samples the same number of real videos as each fake
task-method group. If the real pool is smaller, it shuffles and samples with
replacement. Every detector output has an explicit
version suffix; methods without selectable variants use `.default.csv`.
`--method cnndet` uses the default checkpoint variant; `--method
cnndet:blur_jpg_prob0_1` selects another variant. Each run updates that
version's rows in `results/oob/cnndet.csv` while preserving other versions.
`--all results/` or `--all results/*/*` combines those summaries into the wide `results/auc.csv`;
detector columns use labels such as `CNNDet(blur_jpg_prob0_5)`. The
macro-averaged detector-family × generator-category table is
`results/auc_cate.csv`. Real videos do not appear as a generator method.

## VBench

Run the three VBench dimensions used by FakeParts with the direct runner:

```bash
python fakeparts/evaluation/run_vbench.py \
  --videos FakePartsBench/fake/t2v/wan21 \
  --run-name wan21
```

The summary reports:

| Output label | VBench dimension |
| --- | --- |
| `Cons.` | `subject_consistency` |
| `Flick.` | `temporal_flickering` |
| `Qual.` | `imaging_quality` |

VBench needs a CUDA-enabled PyTorch environment. By default, checkpoints are
cached under `~/.cache/vbench`; `--cache-dir` changes that location. The script
recursively indexes `.mp4` and `.gif` files using symlinks in
`/tmp/fakeparts-vbench/<run-name>` (or `--work-dir`) because VBench expects a
flat input directory. The raw VBench result files and a normalized summary JSON
are written to `--output-dir`. The unified shell entry point defaults this to
`results/eval/vbench/`.

## FVD_W and FVD_FP

Provide one generated-video folder and two real-reference folders:

```bash
python fakeparts/evaluation/run_fvd.py \
  --fake-dir FakePartsBench/fake/t2v/wan21 \
  --webvid-real-dir /path/to/WebVid-10M/videos \
  --fakeparts-real-dir FakePartsBench/real
```

By default, results are saved under `results/vbench/` and
`results/fvd/<run-name>.json`. Pass `--output-dir` to VBench or `--output` to
FVD to choose a different location.

The JSON contains `FVD_W` (generated videos against the WebVid reference) and
`FVD_FP` (the same generated videos against the FakeParts real reference). The
default feature backbone is I3D. By default, each set uses 10,000 distinct
videos, with 16 consecutive frames decoded at resolution 128, seed 0, and the
upstream I3D checkpoint. Override these settings with `--n-samples`,
`--sequence-length`, `--resolution`, `--seed`, `--model`, and `--checkpoint` as
needed. Each source directory must contain at least `--n-samples` supported
videos; the script refuses to duplicate videos to reach the requested count.

The direct runner defaults to `results/fvd/`; the unified shell entry point
defaults to `results/eval/fvd/`. The script creates a persistent symlink index and the upstream loader's video
metadata cache under `/tmp/fakeparts-fvd/<output-stem>` (or `--work-dir`). This
keeps generated cache files out of the dataset. The checkpoint defaults to
`~/.cache/fakeparts/content-debiased-fvd/` and may be downloaded by the upstream
implementation if it is not already present. FVD values depend on the selected
videos, reference set, checkpoint, and sampling settings; the result JSON
records these details for reproducibility.

## Submodules and environments

Initialize the evaluation submodules with:

```bash
git submodule update --init --recursive
```

Install each upstream project's dependencies in an environment suitable for
its model requirements. VBench and FVD are separate implementations and may
require different PyTorch or CUDA package versions; use separate environments
if their dependency requirements conflict.
