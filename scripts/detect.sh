#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --job-name=fakeparts-detect
#SBATCH --output=logs/detection/%x_%j.out
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/detect.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  is_sida=false
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--det-method" || "$previous" == "--model" || "$previous" == "-m" ]]; then
      [[ "${argument,,}" == "sida" ]] && is_sida=true
      break
    fi
    previous="$argument"
  done
  if [[ "$is_sida" == true ]]; then
    sida_python="${FAKEPARTS_PYTHON:-${HOME}/CausalReactBench/.venv/bin/python}"
    if [[ ! -x "$sida_python" ]]; then
      echo "SIDA Python environment is unavailable: $sida_python" >&2
      exit 127
    fi
    exec "$sida_python" "${SCRIPT_DIR}/detect.py" "$@"
  fi
fi

exec "${FAKEPARTS_PYTHON:-python3}" "${SCRIPT_DIR}/detect.py" "$@"
