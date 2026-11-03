#!/usr/bin/env bash

eval_repo_root() {
  local script_path="$1"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/fakeparts/evaluation/run_vbench.py" ]]; then
    printf '%s\n' "${SLURM_SUBMIT_DIR}"
  else
    local script_dir
    script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
    cd -- "${script_dir}/../.." && pwd
  fi
}

eval_python() {
  if [[ -n "${FAKEPARTS_EVAL_PYTHON:-}" ]]; then
    printf '%s\n' "$FAKEPARTS_EVAL_PYTHON"
  elif [[ -x "${HOME}/CausalReactBench/.venv/bin/python" ]]; then
    printf '%s\n' "${HOME}/CausalReactBench/.venv/bin/python"
  else
    printf '%s\n' "${FAKEPARTS_PYTHON:-python3}"
  fi
}

eval_environment() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && type module >/dev/null 2>&1; then
    module load python/3.10.12 uv/0.9.9
  fi
  export PYTHONPATH="${EVAL_ROOT}/.eval_site_packages/py310:${EVAL_ROOT}/fakeparts/evaluation/VBench:${EVAL_ROOT}/fakeparts/evaluation/content-debiased-fvd${PYTHONPATH:+:${PYTHONPATH}}"
}
