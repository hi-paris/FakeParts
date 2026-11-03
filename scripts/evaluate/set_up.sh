#!/usr/bin/env bash
#SBATCH --account=images
#SBATCH --qos=normal
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=fakeparts-eval-setup
#SBATCH --output=evaluate-setup-%j.out
#SBATCH --error=evaluate-setup-%j.err
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/evaluate/set_up.sh" ]]; then
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/scripts/evaluate/set_up.sh"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
EVAL_ROOT="$(eval_repo_root "$SCRIPT_PATH")"

usage() {
  cat <<'EOF'
Usage: set_up.sh [--python PATH] [--uv PATH]

Install the pinned VBench/FVD evaluation dependencies into
.eval_site_packages/py310 using the selected Python environment's PyTorch.

Options:
  --python PATH  Python environment with PyTorch (default: CausalReactBench venv)
  --uv PATH      uv executable (default: uv from PATH)
  -h, --help     Show this help
EOF
}

PYTHON_OVERRIDE=""
UV_BIN="${FAKEPARTS_UV:-uv}"
while (($#)); do
  case "$1" in
    --python) PYTHON_OVERRIDE="$2"; shift 2 ;;
    --uv) UV_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON_OVERRIDE:-$(eval_python)}"
command -v "$PYTHON" >/dev/null 2>&1 || [[ -x "$PYTHON" ]] || {
  echo "Python executable not found: $PYTHON" >&2; exit 127;
}
command -v "$UV_BIN" >/dev/null 2>&1 || [[ -x "$UV_BIN" ]] || {
  echo "uv executable not found: $UV_BIN" >&2; exit 127;
}
EXTRA_SITE="${EVAL_ROOT}/.eval_site_packages/py310"
mkdir -p "$EXTRA_SITE"
eval_environment

"$UV_BIN" pip install --python "$PYTHON" --python-version 3.10 --target "$EXTRA_SITE" --no-deps --only-binary :all: \
  scipy==1.15.3 scikit-image==0.25.2 timm==1.0.30 pyiqa==0.1.16 \
  addict==2.4.0 yapf==0.43.0 lmdb==3.0.0 lazy-loader==0.6 tifffile==2025.5.10

cd "$EVAL_ROOT"
"$PYTHON" - <<'PY'
import torch
from cdfvd.fvd import cdfvd
from vbench import VBench
from vbench.imaging_quality import compute_imaging_quality
from vbench.subject_consistency import compute_subject_consistency
from vbench.temporal_flickering import compute_temporal_flickering

print(f"Evaluation imports ready; torch={torch.__version__}, CUDA runtime={torch.version.cuda}")
PY
