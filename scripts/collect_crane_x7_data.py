# Collect CRANE-X7 cube-lift demonstrations with a scripted expert.
#
# The expert plans end-effector waypoints in Cartesian space using the
# privileged cube position (hover -> descend -> grasp -> lift), converts each
# waypoint to joint angles with the environment's IK helper, and executes them
# with joint position control. The dataset stores those IK joint targets as
# `action` (7 arm joints + 1 gripper), i.e. the same absolute joint-position
# action space the policies use at rollout time.
#
# Episodes are saved in this repo's LeRobot-style layout (parquet + mp4 +
# meta) under the local Hugging Face cache, so training configs can reference
# the dataset by repo_id without uploading it to the Hub:
#
#   python -m scripts.collect_crane_x7_data \
#       --config_path configs/crane_x7/defaults.yaml
#
#   dataset:
#     repo_ids: [local/crane_x7_lift]
#     train_episodes_range: [0, 50]
#     test_episodes_range: [50, 60]

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEFAULT_CONFIG_PATH = Path(
    os.environ.get('CRANE_X7_DEFAULTS', _REPO_ROOT / 'configs/crane_x7/defaults.yaml')
)

from init import init; init()

import numpy as np


def load_defaults_config(config_path):
    import yaml

    with Path(config_path).open('r') as f:
        return yaml.safe_load(f) or {}


def collection_default(config, key, fallback=None):
    collection = config.get('collection', {})
    if key in collection:
        return collection[key]
    environment = config.get('environment', {})
    if key in {'img_size', 'show_viewer', 'visualize_camera', 'camera_view'}:
        return environment.get(key, fallback)
    return fallback


def make_env(args):
    import gymnasium as gym
    import custom_envs  # noqa: F401  (registers CraneX7-v0)
    return gym.make(
        id='CraneX7-v0',
        img_size=args.img_size,
        max_episode_steps=args.max_steps,
        show_viewer=args.show_viewer,
        visualize_camera=args.visualize_camera,
        camera_view=args.camera_view,
    )


class ScriptedLiftExpert:
    '''Cartesian waypoint expert: hover above the cube, descend, close the
    gripper, and lift. Emits absolute joint-position actions via IK.'''

    def __init__(self, env, z_hover=0.20, z_grasp=None, z_lift=0.28,
                 grasp_ee_clearance=0.065, step_size=0.010, reach_tol=0.02,
                 grasp_steps=20, pregrasp=0.45):
        self.env = env.unwrapped
        self.z_hover = z_hover
        self.z_grasp_override = z_grasp
        self.grasp_ee_clearance = grasp_ee_clearance
        self.z_lift = z_lift
        self.step_size = step_size
        self.reach_tol = reach_tol
        self.grasp_steps = grasp_steps
        # Narrow finger opening during the approach: a short closing sweep
        # keeps the cube from being ejected sideways when the fingers close.
        self.pregrasp = pregrasp

    def reset(self):
        self.phase = 'hover'
        self.grasp_counter = 0
        self.virtual_target = self.env.get_ee_pos().copy()
        cube_pos = self.env.get_cube_pos()
        self.cube_xy = cube_pos[:2].copy()
        if self.z_grasp_override is None:
            cube_top = float(cube_pos[2] + self.env.cube.half)
            self.z_grasp = cube_top + self.grasp_ee_clearance
        else:
            self.z_grasp = float(self.z_grasp_override)

    def _waypoint_and_grip(self):
        from custom_envs.crane_x7_env import GRIPPER_CLOSE
        x, y = self.cube_xy
        if self.phase == 'hover':
            return np.array([x, y, self.z_hover]), self.pregrasp
        if self.phase == 'descend':
            return np.array([x, y, self.z_grasp]), self.pregrasp
        if self.phase == 'grasp':
            return np.array([x, y, self.z_grasp]), GRIPPER_CLOSE
        return np.array([x, y, self.z_lift]), GRIPPER_CLOSE  # lift

    def _advance_phase(self, waypoint):
        ee = self.env.get_ee_pos()
        reached = np.linalg.norm(ee - waypoint) < self.reach_tol
        if self.phase == 'hover' and reached:
            self.phase = 'descend'
        elif self.phase == 'descend' and reached:
            self.phase = 'grasp'
        elif self.phase == 'grasp':
            self.grasp_counter += 1
            if self.grasp_counter >= self.grasp_steps:
                self.phase = 'lift'

    def act(self):
        waypoint, grip = self._waypoint_and_grip()
        self._advance_phase(waypoint)
        waypoint, grip = self._waypoint_and_grip()  # phase may have changed
        delta = waypoint - self.virtual_target
        dist = np.linalg.norm(delta)
        if dist > self.step_size:
            delta = delta / dist * self.step_size
        self.virtual_target = self.virtual_target + delta
        return self.env.solve_ik_action(self.virtual_target, grip)


def to_row(obs, action, episode_index, frame_index, fps,
           reward, done, terminated, truncated):
    from PIL import Image
    row = {}
    for k, v in obs.items():
        v = np.asarray(v)
        if 'image' in k:
            v = Image.fromarray(v)
        elif v.ndim == 0:
            v = v.item()
        row[k] = v
    row.update({
        'action': np.asarray(action, dtype=np.float32),
        'episode_index': episode_index,
        'frame_index': frame_index,
        'timestamp': frame_index / fps,
        'next.reward': float(reward),
        'next.done': bool(done),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
    })
    return row


def collect_episode(env, expert, seed, episode_index, fps, settle_steps):
    obs, _ = env.reset(seed=seed)
    if settle_steps > 0:
        obs = env.unwrapped.settle(settle_steps)
    expert.reset()
    rows, success, done, frame_index = [], False, False, 0
    while not done:
        action = expert.act()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        rows.append(to_row(obs, action, episode_index, frame_index, fps,
                           reward, done, terminated, truncated))
        success = success or bool(info.get('success', terminated))
        obs = next_obs
        frame_index += 1
    return rows, success


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config_path', type=str, default=str(_DEFAULT_CONFIG_PATH),
                               help='Path to the CRANE-X7 defaults YAML')
    config_args, _ = config_parser.parse_known_args()
    defaults = load_defaults_config(config_args.config_path)
    camera_views = defaults.get('environment', {}).get('camera_views', {})
    camera_view_choices = tuple(camera_views) or ('right',)

    parser = argparse.ArgumentParser(
        description='Collect CRANE-X7 demonstrations',
        parents=[config_parser],
    )
    parser.add_argument('--repo_id', type=str,
                        default=collection_default(defaults, 'repo_id', 'local/crane_x7_lift'))
    parser.add_argument('--num_episodes', type=int,
                        default=collection_default(defaults, 'num_episodes', 60))
    parser.add_argument('--max_steps', type=int,
                        default=collection_default(defaults, 'max_steps', 300))
    parser.add_argument('--img_size', type=int,
                        default=collection_default(defaults, 'img_size', 128))
    parser.add_argument('--seed', type=int,
                        default=collection_default(defaults, 'seed', 0))
    parser.add_argument('--z_grasp', type=float,
                        default=collection_default(defaults, 'z_grasp', None),
                        help='fixed EE height when grasping; defaults to cube top + clearance')
    parser.add_argument('--grasp_ee_clearance', type=float,
                        default=collection_default(defaults, 'grasp_ee_clearance', 0.065),
                        help='EE height above the measured cube top when --z_grasp is not set')
    parser.add_argument('--z_lift', type=float,
                        default=collection_default(defaults, 'z_lift', 0.28),
                        help='EE height after grasping')
    parser.add_argument('--settle_steps', type=int,
                        default=collection_default(defaults, 'settle_steps', 20),
                        help='raw physics steps after reset before planning/recording')
    parser.add_argument('--show_viewer', action=argparse.BooleanOptionalAction,
                        default=collection_default(defaults, 'show_viewer', False),
                        help='open the Genesis viewer to watch the collection')
    parser.add_argument('--visualize_camera', action=argparse.BooleanOptionalAction,
                        default=collection_default(defaults, 'visualize_camera', False),
                        help='draw the observation camera frustum in the Genesis viewer')
    parser.add_argument('--camera_view', choices=camera_view_choices,
                        default=collection_default(defaults, 'camera_view', 'right'),
                        help='observation camera preset; YAML default is right')
    parser.add_argument('--keep_failures', action=argparse.BooleanOptionalAction,
                        default=collection_default(defaults, 'keep_failures', False),
                        help='also keep unsuccessful episodes')
    parser.add_argument('--max_attempts_factor', type=int,
                        default=collection_default(defaults, 'max_attempts_factor', 3))
    args = parser.parse_args()
    os.environ['CRANE_X7_DEFAULTS'] = str(Path(args.config_path).resolve())

    env = make_env(args)
    expert = ScriptedLiftExpert(
        env,
        z_grasp=args.z_grasp,
        z_lift=args.z_lift,
        grasp_ee_clearance=args.grasp_ee_clearance,
    )
    fps = env.metadata['render_fps']

    all_rows, episode_index, attempt = [], 0, 0
    max_attempts = args.num_episodes * args.max_attempts_factor
    num_success = 0
    while episode_index < args.num_episodes and attempt < max_attempts:
        rows, success = collect_episode(
            env, expert, args.seed + attempt, episode_index, fps, args.settle_steps)
        attempt += 1
        if success or args.keep_failures:
            all_rows.extend(rows)
            episode_index += 1
            num_success += int(success)
            print(f'episode {episode_index}/{args.num_episodes} '
                  f'(attempt {attempt}, steps={len(rows)}, success={success})')
        else:
            print(f'attempt {attempt}: failed, retrying with a new seed')
    env.close()

    if episode_index < args.num_episodes:
        raise RuntimeError(
            f'collected only {episode_index}/{args.num_episodes} episodes after '
            f'{attempt} attempts; tune the expert (e.g. --z_grasp) first'
        )

    from datasets import Dataset as HFDataset
    from robo_utils.dataset.save_utils import save_dataset_pipeline
    from robo_utils.dataset.episode_dataset import DATASETS_HOME

    dataset = HFDataset.from_list(all_rows)
    save_dataset_pipeline(dataset, args.repo_id, fps=fps)

    out_dir = DATASETS_HOME / args.repo_id
    print(f'\nsaved {episode_index} episodes ({num_success} successful, '
          f'{len(all_rows)} frames) to {out_dir}')
    print('use it in a config with:')
    print(f'  dataset:\n    repo_ids: [{args.repo_id}]\n'
          f'    train_episodes_range: [0, {int(episode_index * 5 / 6)}]\n'
          f'    test_episodes_range: [{int(episode_index * 5 / 6)}, {episode_index}]')


if __name__ == '__main__':
    main()
