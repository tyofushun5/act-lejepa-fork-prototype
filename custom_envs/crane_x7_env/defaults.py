'''YAML-backed defaults for the CRANE-X7 environment.'''
import os
from functools import lru_cache
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = _REPO_ROOT / 'configs/crane_x7/defaults.yaml'


@lru_cache(maxsize=None)
def load_crane_x7_defaults(path=None):
    config_path = Path(path or os.environ.get('CRANE_X7_DEFAULTS') or DEFAULTS_PATH)
    with config_path.open('r') as f:
        return yaml.safe_load(f) or {}


def get_env_defaults():
    return load_crane_x7_defaults().get('environment', {})


def get_robot_defaults():
    return load_crane_x7_defaults().get('robot', {})
