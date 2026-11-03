#!/usr/bin/env bash
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=fakeparts-generator-setup
#SBATCH --output=fakeparts-generator-setup-%j.out
#SBATCH --error=fakeparts-generator-setup-%j.err
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/set_up_generator.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "Environment Modules is unavailable in this Slurm job; load python/3.10.12 and uv/0.9.9." >&2
    exit 127
  fi
  module purge
  module load python/3.10.12 uv/0.9.9
fi
if [[ ! -f "${SCRIPT_DIR}/set_up_generator.py" ]]; then
  echo "Cannot locate set_up_generator.py. Submit from the repository root (sbatch scripts/set_up_generator.sh) or run this script with bash." >&2
  exit 2
fi
exec "${FAKEPARTS_PYTHON:-python3}" "$SCRIPT_DIR/set_up_generator.py" "$@"
