#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

REPO_ID="${REPO_ID:-local/crane_x7_lift}"
NUM_EPISODES="${NUM_EPISODES:-60}"
SHOW_VIEWER="${SHOW_VIEWER:-1}"
KEEP_FAILURES="${KEEP_FAILURES:-0}"

cmd=(
  "${PYTHON_BIN}" -m scripts.collect_crane_x7_data
  --repo_id "${REPO_ID}"
  --num_episodes "${NUM_EPISODES}"
)

if [[ "${SHOW_VIEWER}" != "0" ]]; then
  cmd+=(--show_viewer)
fi

if [[ "${KEEP_FAILURES}" != "0" ]]; then
  cmd+=(--keep_failures)
fi

cmd+=("$@")

echo "Running data collection"
echo "repo_id=${REPO_ID} num_episodes=${NUM_EPISODES} show_viewer=${SHOW_VIEWER} keep_failures=${KEEP_FAILURES}"
"${cmd[@]}"
