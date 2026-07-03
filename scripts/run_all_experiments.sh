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

tasks=(pusht metaworld mani_skill)
models=(act act-jepa act-lejepa)

for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    cfg="configs/${task}/${model}.yaml"
    if [[ ! -f "${cfg}" ]]; then
      echo "Missing config: ${cfg}" >&2
      exit 1
    fi

    echo "Running ${cfg}"
    "${PYTHON_BIN}" -m scripts.train --config_path "${cfg}"
  done
done
