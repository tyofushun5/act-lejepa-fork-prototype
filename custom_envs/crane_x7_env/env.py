'''CRANE-X7 cube-lift environment (Genesis) in this repo's gymnasium convention.

Structure follows `samples/adrobo-CRANE-X7-main/simulation`, with entities and
Genesis setup in `custom_envs/crane_x7_env/{entity,config}`. Differences from
that reference (which targets DreamerV2):

- Standard single-env gymnasium API: 5-tuple step, no internal auto-reset
  (truncation is handled by gymnasium's TimeLimit via `max_episode_steps`).
- Dict observations following this repo's convention:
  `observation.state` (9 joint positions), `observation.environment_state`
  (cube xyz + end-effector xyz), `observation.image`, `task_index`.
- Continuous absolute joint-position actions (7 arm joints + 1 gripper value
  mirrored to both fingers) instead of Discrete(8) end-effector deltas.
  Datasets collected by `scripts/collect_crane_x7_data.py` store the scripted
  expert's IK joint targets in this same action space.
- `terminated` == success (cube lifted above `success_height`), matching the
  evaluator's `solved %` convention.
- `scene.step()` is called `substeps` times per control step (the reference
  passed `substeps` as `scene.step()`'s first positional argument, which is
  actually `update_visualizer` and stepped physics only once).
'''
import numpy as np
import gymnasium as gym

from .config import GenesisConfig
from .entity.camera import ObsCamera
from .entity.crane_x7 import CraneX7
from .entity.cube import Cube
from .entity.table import Table
from .entity.workspace import Workspace


class CraneX7Env(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        render_mode='rgb_array',
        img_size=128,
        cube_size=0.025,
        cube_margin=0.03,
        success_height=None,
        substeps=40,
        device='cpu',
        show_viewer=False,
        urdf_path=None,
        cam_pos=(1.0, 1.0, 0.10),
        cam_lookat=(0.200, 0.0, 0.10),
        cam_fov=30.0,
    ):
        super().__init__()
        import genesis as gs

        self.render_mode = render_mode
        self.img_size = int(img_size)
        self.cube_margin = float(cube_margin)
        self.substeps = int(substeps)

        self.genesis_cfg = GenesisConfig(device=device, show_viewer=show_viewer)
        self.scene = self.genesis_cfg.gs_init()

        self.workspace = Workspace()
        # Lift target: halfway up the workspace, as in the reference.
        self.success_height = float(
            success_height if success_height is not None
            else self.workspace.workspace_min[2]
            + 0.5 * (self.workspace.workspace_max[2] - self.workspace.workspace_min[2])
        )

        self.table = Table(scene=self.scene)
        self.table.create()
        # Floor plane below the table, matching the sample environment. The
        # table collision box itself provides the tabletop at z=0.
        self.scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -self.table.table_height)))

        self.robot = CraneX7(scene=self.scene, urdf_path=urdf_path, workspace=self.workspace)
        self.robot.create()
        self.cube = Cube(scene=self.scene, size=cube_size)
        self.cube.create()
        self.camera = ObsCamera(
            scene=self.scene, res=(self.img_size, self.img_size),
            pos=cam_pos, lookat=cam_lookat, fov=cam_fov,
        )
        self.camera.create()

        self.scene.build()
        self.robot.setup()

        self.observation_space = gym.spaces.Dict({
            'observation.state': gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
            'observation.environment_state': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            'observation.image': gym.spaces.Box(0, 255, (self.img_size, self.img_size, 3), np.uint8),
            'task_index': gym.spaces.Box(0, np.inf, (), int),
        })
        low = self.robot.command_lower
        high = self.robot.command_upper
        self.action_space = gym.spaces.Box(
            np.concatenate([low[:7], [low[7]]]).astype(np.float32),
            np.concatenate([high[:7], [high[7]]]).astype(np.float32),
            (8,), np.float32,
        )

    # ------------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()

        low = self.workspace.workspace_min[:2] + self.cube_margin
        high = self.workspace.workspace_max[:2] - self.cube_margin
        xy = self.np_random.uniform(low, high)
        self.cube.reset(np.array([xy[0], xy[1], self.cube.half + 1e-3]))

        for _ in range(10):
            self.scene.step()

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape == (8,), f'expected 8-dim joint action, got {action.shape}'
        qpos_target = np.empty(9, dtype=np.float64)
        qpos_target[:7] = action[:7]
        qpos_target[7] = qpos_target[8] = action[7]  # mirror gripper fingers
        self.robot.control_qpos(qpos_target)

        for _ in range(self.substeps):
            self.scene.step()

        cube_pos = self.cube.get_pos()
        success = bool(cube_pos[2] >= self.success_height)
        reward = float(success)
        terminated = success
        truncated = False  # TimeLimit wrapper handles episode truncation
        info = {'success': success, 'cube_height': float(cube_pos[2])}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self.camera.get_image()

    def close(self):
        pass

    # ------------------------------------------------------------- helpers

    def settle(self, steps):
        for _ in range(int(steps)):
            self.scene.step()
        return self._get_obs()

    def _get_obs(self):
        return {
            'observation.state': self.robot.get_qpos().astype(np.float32),
            'observation.environment_state': np.concatenate(
                [self.cube.get_pos(), self.robot.get_ee_pos()]).astype(np.float32),
            'observation.image': self.render(),
            'task_index': np.asarray(0),
        }

    def get_cube_pos(self):
        return self.cube.get_pos()

    def get_ee_pos(self):
        return self.robot.get_ee_pos()

    def solve_ik_action(self, target_pos, gripper):
        return self.robot.solve_ik(target_pos, gripper)


def make_crane_x7_env(
    env_name=None,  # unused; single task ('lift-cube')
    render_mode='rgb_array',
    img_size=128,
    **kwargs,
):
    return CraneX7Env(render_mode=render_mode, img_size=img_size, **kwargs)


gym.register(id='CraneX7-v0', entry_point=make_crane_x7_env)
