#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <env_name> [evaluate.py args...]" >&2
  exit 1
fi

task="$1"
shift
eval_args=("$@")

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

if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/act-jepa-matplotlib"
  mkdir -p "${MPLCONFIGDIR}"
fi

is_enabled() {
  case "${1:-}" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF) return 1 ;;
    *) return 0 ;;
  esac
}

run_train="${RUN_TRAIN:-1}"
run_eval="${RUN_EVAL:-1}"
log_eval_wandb="${LOG_EVAL_WANDB:-1}"

if ! is_enabled "${run_train}" && ! is_enabled "${run_eval}"; then
  echo "both RUN_TRAIN and RUN_EVAL are disabled" >&2
  exit 1
fi

models_string="${MODELS:-act act-jepa act-lejepa}"
models_string="${models_string//,/ }"
read -r -a models <<< "${models_string}"

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "no models specified" >&2
  exit 1
fi

tmp_configs=()
cleanup() {
  for cfg in "${tmp_configs[@]:-}"; do
    [[ -f "${cfg}" ]] && rm -f "${cfg}"
  done
}
trap cleanup EXIT

make_crane_eval_config() {
  local base_config="$1"
  local out_config="$2"
  local camera_view="$3"

  "${PYTHON_BIN}" - "$base_config" "$out_config" "$camera_view" \
    "${EVAL_NUM_EPISODES:-${NUM_EPISODES:-}}" \
    "${EVAL_MAX_EPISODE_STEPS:-${MAX_EPISODE_STEPS:-}}" \
    "${SHOW_VIEWER:-0}" \
    "${SHOW_CAMERAS:-}" <<'PY'
from pathlib import Path
import sys
import yaml

base_config = Path(sys.argv[1]).resolve()
out_config = Path(sys.argv[2]).resolve()
camera_view = sys.argv[3]
num_episodes = sys.argv[4]
max_episode_steps = sys.argv[5]
show_viewer = sys.argv[6]
show_cameras = sys.argv[7]


def as_bool(value):
    return str(value).lower() not in {"", "0", "false", "no", "off"}


def make_loader(base_dir):
    class IncludeLoader(yaml.SafeLoader):
        pass

    def include(loader, node):
        path = Path(loader.construct_scalar(node))
        if not path.is_absolute():
            path = base_dir / path
        with path.open("r") as f:
            return yaml.load(f, IncludeLoader)

    IncludeLoader.add_constructor("!inc", include)
    return IncludeLoader


with base_config.open("r") as f:
    config = yaml.load(f, Loader=make_loader(base_config.parent))

env = config.setdefault("env", {})
env_kwargs = env.setdefault("env_kwargs", {})
env_kwargs["camera_view"] = camera_view
env_kwargs["show_viewer"] = as_bool(show_viewer)

if show_cameras != "":
    env_kwargs["show_cameras"] = as_bool(show_cameras)
if num_episodes != "":
    env["num_episodes"] = int(num_episodes)
if max_episode_steps != "":
    env_kwargs["max_episode_steps"] = int(max_episode_steps)

with out_config.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
}

run_evaluation() {
  local cfg="$1"
  local label="$2"
  local wandb_suffix="$3"
  local wandb_run_suffix="${4:-}"

  echo "Evaluating ${label}"
  cmd=("${PYTHON_BIN}" -m scripts.evaluate --config_path "${cfg}")
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    cmd+=(--checkpoint_path "${CHECKPOINT_PATH}")
  fi
  if is_enabled "${log_eval_wandb}"; then
    local wandb_run_name="${task}/${model} eval"
    if [[ -n "${wandb_run_suffix}" ]]; then
      wandb_run_name="${wandb_run_name} ${wandb_run_suffix}"
    fi
    cmd+=(
      --wandb
      --wandb_prefix "Eval/${task}/${wandb_suffix}"
      --wandb_video_prefix "${task}/${wandb_suffix}"
      --wandb_run_name "${wandb_run_name}"
      --wandb_group "${task}/${model}"
    )
  fi
  cmd+=("${eval_args[@]}")
  "${cmd[@]}"
}

run_crane_evaluation() {
  local cfg="$1"

  # CRANE-X7 defaults to the training view plus two held-out camera presets.
  # Override with CAMERA_VIEWS=left,front or CAMERA_VIEW=left.
  views_string="${CAMERA_VIEWS:-${CAMERA_VIEW:-right,left,front}}"
  views_string="${views_string//,/ }"
  read -r -a views <<< "${views_string}"

  for view in "${views[@]}"; do
    [[ -z "${view}" ]] && continue

    tmp_config="$(mktemp "/tmp/act-jepa-crane-x7-${view}.XXXXXX.yaml")"
    tmp_configs+=("${tmp_config}")
    make_crane_eval_config "${cfg}" "${tmp_config}" "${view}"
    run_evaluation "${tmp_config}" "${cfg} camera_view=${view}" "${model}/${view}" "${view}"
  done
}

for model in "${models[@]}"; do
  [[ -z "${model}" ]] && continue

  cfg="configs/${task}/${model}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" >&2
    exit 1
  fi

  if is_enabled "${run_train}"; then
    echo "Training ${cfg}"
    "${PYTHON_BIN}" -m scripts.train --config_path "${cfg}"
  fi

  if is_enabled "${run_eval}"; then
    if [[ "${task}" == "crane_x7" ]]; then
      run_crane_evaluation "${cfg}"
    else
      run_evaluation "${cfg}" "${cfg}" "${model}"
    fi
  fi
done
