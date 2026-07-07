#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_EVAL="${RUN_EVAL:-0}"
exec "${SCRIPT_DIR}/run_env_experiments.sh" crane_x7 "$@"
