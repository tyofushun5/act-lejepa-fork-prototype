'''YAML-backed defaults for the CRANE-X7 environment.'''
from copy import deepcopy
import os
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = _REPO_ROOT / 'configs/crane_x7/defaults.yaml'


def _resolve_defaults_path(path=None):
    return Path(path or os.environ.get('CRANE_X7_DEFAULTS') or DEFAULTS_PATH).expanduser().resolve()


def load_crane_x7_defaults(path=None):
    config_path = _resolve_defaults_path(path)
    with config_path.open('r') as f:
        return deepcopy(yaml.safe_load(f) or {})


def get_env_defaults():
    return load_crane_x7_defaults().get('environment', {})


def get_robot_defaults():
    return load_crane_x7_defaults().get('robot', {})
