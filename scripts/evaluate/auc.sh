#!/usr/bin/env bash
#SBATCH --account=images
#SBATCH --qos=normal
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=fakeparts-auc
#SBATCH --output=evaluate-auc-%j.out
#SBATCH --error=evaluate-auc-%j.err
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/evaluate/auc.sh" ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/evaluate/auc.sh"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
EVAL_ROOT="$(eval_repo_root "$SCRIPT_PATH")"

usage() {
  cat <<'EOF'
Usage:
  auc.sh --method DETECTOR[:VERSION] [--results-root DIR] [--seed N]
  auc.sh --all [RESULTS_ROOT_OR_DETECTOR_DIR ...]

--method balances each fake task-method CSV against scored real_*.csv shards and writes one detector summary.
--all recomputes per-method AUCs and writes results/auc.csv and results/auc_cate.csv.
     With no directories, it scans results/{oob,fom,vlm,dib}/.

Options:
--method ID[:VERSION]
                   Detector ID and optional checkpoint version, e.g.
                   cnndet or cnndet:blur_jpg_prob0_1
--all DIR...       Optional results root/family/detector directories; defaults to results/{oob,fom,vlm,dib}/
--aggregate DIR... Optional paths with existing per-detector AUC CSVs; same defaults; does not recompute scores
  --results-root DIR Results root for --method (default: results)
  --seed N           Real-control sampling seed (default: 0)
  -h, --help         Show this help
EOF
}

MODE=""
DETECTOR=""
RESULTS_ROOT=""
ALL_DIRS=()
SEED="0"
while (($#)); do
  case "$1" in
    --method) MODE=method; DETECTOR="$2"; shift 2 ;;
    --all) MODE=all; shift ;;
    --aggregate) MODE=aggregate; shift ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ "$MODE" == all || "$MODE" == aggregate ]]; then
        ALL_DIRS+=("$1")
        shift
      else
        echo "Unknown option: $1" >&2; usage >&2; exit 2
      fi
      ;;
  esac
done
[[ -n "$MODE" ]] || { echo "Choose --method or --all" >&2; usage >&2; exit 2; }

PYTHON="$(eval_python)"
eval_environment
if [[ "$MODE" == method ]]; then
  [[ -n "$DETECTOR" ]] || { echo "--method requires a detector ID" >&2; usage >&2; exit 2; }
  ARGS=(--method "$DETECTOR" --seed "$SEED")
  [[ -n "$RESULTS_ROOT" ]] && ARGS+=(--results-root "$RESULTS_ROOT")
elif [[ "$MODE" == aggregate ]]; then
  ARGS=(--aggregate)
  ((${#ALL_DIRS[@]})) && ARGS+=("${ALL_DIRS[@]}")
else
  ARGS=(--all)
  ((${#ALL_DIRS[@]})) && ARGS+=("${ALL_DIRS[@]}")
fi
cd "$EVAL_ROOT"
exec "$PYTHON" scripts/evaluate/auc.py "${ARGS[@]}"
