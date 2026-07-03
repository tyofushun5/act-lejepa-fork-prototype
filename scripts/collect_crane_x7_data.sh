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

CONFIG_PATH="${CONFIG_PATH:-configs/crane_x7/defaults.yaml}"

cmd=(
  "${PYTHON_BIN}" -m scripts.collect_crane_x7_data
  --config_path "${CONFIG_PATH}"
)

if [[ -n "${REPO_ID:-}" ]]; then
  cmd+=(--repo_id "${REPO_ID}")
fi

if [[ -n "${NUM_EPISODES:-}" ]]; then
  cmd+=(--num_episodes "${NUM_EPISODES}")
fi

if [[ -n "${SHOW_VIEWER:-}" ]]; then
  if [[ "${SHOW_VIEWER}" != "0" ]]; then
    cmd+=(--show_viewer)
  else
    cmd+=(--no-show_viewer)
  fi
fi

CAMERA_VIS_FLAG="${SHOW_CAMERAS:-${VISUALIZE_CAMERA:-}}"
if [[ -n "${CAMERA_VIS_FLAG}" ]]; then
  if [[ "${CAMERA_VIS_FLAG}" != "0" ]]; then
    cmd+=(--show_cameras)
  else
    cmd+=(--no-show_cameras)
  fi
fi

if [[ -n "${CAMERA_VIEW:-}" ]]; then
  cmd+=(--camera_view "${CAMERA_VIEW}")
fi

if [[ -n "${KEEP_FAILURES:-}" ]]; then
  if [[ "${KEEP_FAILURES}" != "0" ]]; then
    cmd+=(--keep_failures)
  else
    cmd+=(--no-keep_failures)
  fi
fi

cmd+=("$@")

echo "Running data collection"
echo "config_path=${CONFIG_PATH}"
"${cmd[@]}"
