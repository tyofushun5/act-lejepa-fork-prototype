#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

tasks=(pusht metaworld mani_skill crane_x7)

for task in "${tasks[@]}"; do
  echo "Running ${task} experiments"
  "${SCRIPT_DIR}/run_env_experiments.sh" "${task}" "$@"
done
