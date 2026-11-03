#!/usr/bin/env bash
#SBATCH --account=images
#SBATCH --qos=normal
#SBATCH --partition=H100,L40S,audible,A100,RTX6000PRO,A40,3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=fakeparts-fvd
#SBATCH --output=evaluate-fvd-%j.out
#SBATCH --error=evaluate-fvd-%j.err
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/evaluate/fvd.sh" ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/evaluate/fvd.sh"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
EVAL_ROOT="$(eval_repo_root "$SCRIPT_PATH")"

usage() {
  cat <<'EOF'
Usage: fvd.sh --input-jsonl FILE [options]

Compute FVD_FP and, by default, FVD_W from fake videos listed in a JSONL.
Submit directly with: sbatch scripts/evaluate/fvd.sh --input-jsonl FILE

Options:
  --input-jsonl FILE       JSONL whose rows contain file_name and label=fake
  --dataset-root DIR       Root for relative paths (default: auto-detect FakePartsBench)
  --real-dir DIR           FakeParts real-reference videos (default: FakePartsBench/real)
  --webvid-dataset ID      WebVid reference (default: TempoFunk/webvid-10M)
  --webvid-split NAME      WebVid split (default: validation)
  --webvid-download-workers N (default: 16)
  --fakeparts-only         Compute only FVD_FP; do not fetch WebVid videos
  --n-samples N            Videos per set (default: number of fake JSONL rows)
  --run-name NAME          Output stem (default: JSONL name + timestamp)
  --output FILE            Default: results/eval/fvd/<run-name>.json
  --model i3d|videomae     Feature backbone (default: i3d)
  --batch-size N           Video feature batch size (default: 8)
  --num-workers N          Data loader workers (default: 4)
  --device DEVICE          Torch device (default: cuda)
  --half-precision         Use half precision
  -h, --help               Show this help
EOF
}

INPUT_JSONL=""
DATASET_ROOT=""
REAL_DIR="${EVAL_ROOT}/FakePartsBench/real"
WEBVID_DATASET="TempoFunk/webvid-10M"
WEBVID_SPLIT=validation
WEBVID_WORKERS=16
FAKEPARTS_ONLY=0
N_SAMPLES=""
RUN_NAME=""
OUTPUT=""
MODEL=i3d
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE=cuda
HALF_PRECISION=0
while (($#)); do
  case "$1" in
    --input-jsonl) INPUT_JSONL="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --real-dir) REAL_DIR="$2"; shift 2 ;;
    --webvid-dataset) WEBVID_DATASET="$2"; shift 2 ;;
    --webvid-split) WEBVID_SPLIT="$2"; shift 2 ;;
    --webvid-download-workers) WEBVID_WORKERS="$2"; shift 2 ;;
    --fakeparts-only) FAKEPARTS_ONLY=1; shift ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --half-precision) HALF_PRECISION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$INPUT_JSONL" ]] || { echo "--input-jsonl is required" >&2; usage >&2; exit 2; }
[[ "$INPUT_JSONL" = /* ]] || INPUT_JSONL="${EVAL_ROOT}/${INPUT_JSONL}"
[[ -f "$INPUT_JSONL" ]] || { echo "JSONL not found: $INPUT_JSONL" >&2; exit 2; }
[[ "$REAL_DIR" = /* ]] || REAL_DIR="${EVAL_ROOT}/${REAL_DIR}"
[[ -n "$RUN_NAME" ]] || RUN_NAME="$(basename "${INPUT_JSONL%.*}")_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-${EVAL_ROOT}/results/eval/fvd/${RUN_NAME}.json}"
[[ "$OUTPUT" = /* ]] || OUTPUT="${EVAL_ROOT}/${OUTPUT}"
[[ -n "$DATASET_ROOT" ]] || DATASET_ROOT=""
[[ -n "$DATASET_ROOT" && "$DATASET_ROOT" != /* ]] && DATASET_ROOT="${EVAL_ROOT}/${DATASET_ROOT}"

PYTHON="$(eval_python)"
eval_environment
INPUT_KEY="$(sha256sum "$INPUT_JSONL" | cut -c1-12)"
VIDEO_DIR="${EVAL_ROOT}/results/eval/inputs/$(basename "${INPUT_JSONL%.*}")_${INPUT_KEY}_fake"
STAGE_ARGS=(--input-jsonl "$INPUT_JSONL" --output-dir "$VIDEO_DIR" --label fake)
[[ -n "$DATASET_ROOT" ]] && STAGE_ARGS+=(--input-root "$DATASET_ROOT")
VIDEO_COUNT="$("$PYTHON" "${SCRIPT_DIR}/jsonl_to_video_dir.py" "${STAGE_ARGS[@]}")"
N_SAMPLES="${N_SAMPLES:-$VIDEO_COUNT}"
WORK_DIR="/tmp/fakeparts-fvd-${SLURM_JOB_ID:-$$}-${RUN_NAME}"

RUN_ARGS=(--fake-dir "$VIDEO_DIR" --fakeparts-real-dir "$REAL_DIR"
  --n-samples "$N_SAMPLES" --run-name "$RUN_NAME" --output "$OUTPUT"
  --work-dir "$WORK_DIR" --model "$MODEL" --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS" --device "$DEVICE")
if (( ! FAKEPARTS_ONLY )); then
  RUN_ARGS+=(--webvid-dataset "$WEBVID_DATASET" --webvid-split "$WEBVID_SPLIT"
    --webvid-download-workers "$WEBVID_WORKERS")
fi
((HALF_PRECISION)) && RUN_ARGS+=(--half-precision)
cd "$EVAL_ROOT"
echo "FVD input: $INPUT_JSONL ($VIDEO_COUNT fake videos; using $N_SAMPLES)"
exec "$PYTHON" fakeparts/evaluation/run_fvd.py "${RUN_ARGS[@]}"
