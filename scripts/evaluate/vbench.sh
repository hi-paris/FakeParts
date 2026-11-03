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
#SBATCH --job-name=fakeparts-vbench
#SBATCH --output=evaluate-vbench-%j.out
#SBATCH --error=evaluate-vbench-%j.err
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/evaluate/vbench.sh" ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/evaluate/vbench.sh"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
EVAL_ROOT="$(eval_repo_root "$SCRIPT_PATH")"

usage() {
  cat <<'EOF'
Usage: vbench.sh --input-jsonl FILE [options]

Evaluate the fake videos listed in a JSONL with VBench Cons., Flick., and Qual.
Submit directly with: sbatch scripts/evaluate/vbench.sh --input-jsonl FILE

Options:
  --input-jsonl FILE       JSONL whose rows contain file_name and label=fake
  --dataset-root DIR       Root for relative paths (default: auto-detect FakePartsBench)
  --run-name NAME          Output stem (default: JSONL name + timestamp)
  --output-dir DIR         Default: results/eval/vbench
  --gpu ID                 CUDA device index (default: 0)
  --master-port PORT       Distributed port (default: 29500)
  --cache-dir DIR          VBench model cache (default: ~/.cache/vbench)
  --local-checkpoints      Use VBench local-checkpoint mode
  -h, --help               Show this help
EOF
}

INPUT_JSONL=""
DATASET_ROOT=""
RUN_NAME=""
OUTPUT_DIR="${EVAL_ROOT}/results/eval/vbench"
GPU=0
MASTER_PORT=29500
CACHE_DIR="${HOME}/.cache/vbench"
LOCAL_CHECKPOINTS=0
while (($#)); do
  case "$1" in
    --input-jsonl) INPUT_JSONL="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --local-checkpoints) LOCAL_CHECKPOINTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$INPUT_JSONL" ]] || { echo "--input-jsonl is required" >&2; usage >&2; exit 2; }
[[ "$INPUT_JSONL" = /* ]] || INPUT_JSONL="${EVAL_ROOT}/${INPUT_JSONL}"
[[ -f "$INPUT_JSONL" ]] || { echo "JSONL not found: $INPUT_JSONL" >&2; exit 2; }
[[ -n "$RUN_NAME" ]] || RUN_NAME="$(basename "${INPUT_JSONL%.*}")_$(date -u +%Y%m%d_%H%M%S)"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="${EVAL_ROOT}/${OUTPUT_DIR}"
[[ "$CACHE_DIR" = /* ]] || CACHE_DIR="${EVAL_ROOT}/${CACHE_DIR}"
[[ -n "$DATASET_ROOT" ]] || DATASET_ROOT=""
[[ -n "$DATASET_ROOT" && "$DATASET_ROOT" != /* ]] && DATASET_ROOT="${EVAL_ROOT}/${DATASET_ROOT}"

PYTHON="$(eval_python)"
eval_environment
INPUT_KEY="$(sha256sum "$INPUT_JSONL" | cut -c1-12)"
VIDEO_DIR="${EVAL_ROOT}/results/eval/inputs/$(basename "${INPUT_JSONL%.*}")_${INPUT_KEY}_fake"
STAGE_ARGS=(--input-jsonl "$INPUT_JSONL" --output-dir "$VIDEO_DIR" --label fake)
[[ -n "$DATASET_ROOT" ]] && STAGE_ARGS+=(--input-root "$DATASET_ROOT")
VIDEO_COUNT="$("$PYTHON" "${SCRIPT_DIR}/jsonl_to_video_dir.py" "${STAGE_ARGS[@]}")"

RUN_ARGS=(--videos "$VIDEO_DIR" --output-dir "$OUTPUT_DIR" --run-name "$RUN_NAME"
  --gpu "$GPU" --master-port "$MASTER_PORT" --cache-dir "$CACHE_DIR")
((LOCAL_CHECKPOINTS)) && RUN_ARGS+=(--local-checkpoints)
cd "$EVAL_ROOT"
echo "VBench input: $INPUT_JSONL ($VIDEO_COUNT fake videos)"
exec "$PYTHON" fakeparts/evaluation/run_vbench.py "${RUN_ARGS[@]}"
