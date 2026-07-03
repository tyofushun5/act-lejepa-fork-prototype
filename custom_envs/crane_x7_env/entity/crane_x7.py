'''CRANE-X7 robot entity.

Ported from `samples/adrobo-CRANE-X7-main/simulation/entity/crane_x7.py` with
the following fixes:

- URDF `package://crane_x7_description/` URIs are resolved to absolute paths
  before loading. Genesis' strict URDF parser cannot resolve `package://`
  relative to the file location, silently falling back to a legacy parser and
  dropping collision meshes (which made grasping impossible).
- Deprecated Genesis APIs replaced (`joint.dof_idx_local` ->
  `joint.dofs_idx_local`, `link.pose.p` hack -> `link.get_pos()`).
- One consistent gripper convention: positive angle = open (URDF upper limit
  1.571). The URDF lower limit is a squeeze target (-0.087), but policy and
  scripted-expert commands stop at 0.0 to avoid driving through the cube.
  The reference mixed two contradictory conventions between its action handler
  and its IK helper.
- Absolute joint-position control API (used both by the scripted expert and
  by policies at rollout time) instead of the reference's Discrete(8) deltas.
'''
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..defaults import get_robot_defaults
from .entity import Entity
from .workspace import Workspace

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_URDF = _REPO_ROOT / 'samples/adrobo-CRANE-X7-main/crane_x7_description/urdf/crane_x7.urdf'

ARM_JOINT_NAMES = [
    'crane_x7_shoulder_fixed_part_pan_joint',
    'crane_x7_shoulder_revolute_part_tilt_joint',
    'crane_x7_upper_arm_revolute_part_twist_joint',
    'crane_x7_upper_arm_revolute_part_rotate_joint',
    'crane_x7_lower_arm_fixed_part_joint',
    'crane_x7_lower_arm_revolute_part_joint',
    'crane_x7_wrist_joint',
]
GRIPPER_JOINT_NAMES = [
    'crane_x7_gripper_finger_a_joint',
    'crane_x7_gripper_finger_b_joint',
]
EE_LINK_NAME = 'crane_x7_gripper_base_link'

# Rest pose (7 arm + 2 fingers), from the reference.
INIT_QPOS = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, -np.pi / 2, 0.0,
                      0.9, 0.9], dtype=np.float64)
# Top-down end-effector orientation for IK (wxyz).
EE_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

_ROBOT_DEFAULTS = get_robot_defaults()


def _array_default(name, fallback, size):
    values = np.asarray(_ROBOT_DEFAULTS.get(name, fallback), dtype=np.float64).reshape(-1)
    if values.shape != (size,):
        raise ValueError(f'robot.{name} must contain {size} values, got {values.shape[0]}')
    return values


GRIPPER_OPEN = float(_ROBOT_DEFAULTS.get('gripper_open', 0.9))  # positive = open
# Stop at contact instead of commanding the URDF squeeze limit (-0.087), which
# can drive the fingers visually through the cube during scripted collection.
GRIPPER_CLOSE = float(_ROBOT_DEFAULTS.get('gripper_close', 0.0))
GRIPPER_FORCE_LIMIT = float(_ROBOT_DEFAULTS.get('gripper_force_limit', 0.9))
GRIPPER_KP = float(_ROBOT_DEFAULTS.get('gripper_kp', 2.75))
GRIPPER_KV = float(_ROBOT_DEFAULTS.get('gripper_kv', 0.275))
ARM_KP = _array_default('arm_kp', [3520, 3520, 2640, 2640, 1760, 1760, 1760], 7)
ARM_KV = _array_default('arm_kv', [352, 352, 264, 264, 176, 176, 176], 7)

MAX_JOINT_DELTA_FROM_REST = 1.2  # rad, IK sanity clip (from the reference)
MAX_IK_JOINT_STEP = float(_ROBOT_DEFAULTS.get('max_ik_joint_step', 0.10))
MAX_GRIPPER_JOINT_STEP = float(_ROBOT_DEFAULTS.get('max_gripper_joint_step', 0.05))
GRIPPER_COLLISION_LINKS = {
    'crane_x7_gripper_base_link',
    'crane_x7_gripper_finger_a_link',
    'crane_x7_gripper_finger_b_link',
}
GRIPPER_VISUAL_COLLISION_MESHES = {
    'wide_two_finger_gripper_actuator.stl',
    'wide_two_finger_gripper_finger_a.stl',
    'wide_two_finger_gripper_finger_b.stl',
}


def resolve_urdf(urdf_path=None) -> str:
    '''Rewrite `package://crane_x7_description/` URIs to absolute paths.

    The resolved copy is cached next to the original URDF so Genesis' strict
    parser (which loads collision meshes correctly) can be used. Only the
    gripper collision meshes are swapped to their visual mesh counterparts so
    contact around the cube matches what is displayed without making the whole
    robot use heavy visual meshes for collision.
    '''
    urdf_path = Path(urdf_path or os.environ.get('CRANE_X7_URDF') or _DEFAULT_URDF)
    if not urdf_path.exists():
        raise FileNotFoundError(
            f'CRANE-X7 URDF not found: {urdf_path}. Fetch the crane_x7_description '
            'submodule under samples/adrobo-CRANE-X7-main or set CRANE_X7_URDF.'
        )
    pkg_uri = 'package://crane_x7_description/'
    description_root = urdf_path.parent.parent
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall('link'):
        link_name = link.get('name')
        if link_name not in GRIPPER_COLLISION_LINKS:
            for collision in list(link.findall('collision')):
                link.remove(collision)
            continue

        for mesh in link.findall('./collision/geometry/mesh'):
            filename = mesh.get('filename', '')
            mesh_name = Path(filename).name
            if mesh_name in GRIPPER_VISUAL_COLLISION_MESHES:
                filename = filename.replace('meshes/collision/', 'meshes/visual/')
            if filename.startswith(pkg_uri):
                filename = filename.replace(pkg_uri, f'{description_root.as_posix()}/', 1)
            mesh.set('filename', filename)

    for mesh in root.findall('.//visual/geometry/mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith(pkg_uri):
            mesh.set('filename', filename.replace(pkg_uri, f'{description_root.as_posix()}/', 1))

    resolved_text = ET.tostring(root, encoding='unicode')
    resolved_path = urdf_path.with_name(urdf_path.stem + '_resolved_gripper_only_high_precision.urdf')
    if not resolved_path.exists() or resolved_path.read_text() != resolved_text:
        resolved_path.write_text(resolved_text)
    return str(resolved_path)


class CraneX7(Entity):
    def __init__(self, scene=None, surface=None, urdf_path=None, workspace=None):
        super().__init__(scene=scene, surface=surface)
        self.urdf_path = resolve_urdf(urdf_path)
        self.workspace = workspace or Workspace()

        self.entity = None
        self.ee_link = None
        self.arm_dofs = None
        self.gripper_dofs = None
        self.all_dofs = None
        self.dof_lower = None
        self.dof_upper = None
        self.command_lower = None
        self.command_upper = None
        self.last_ik_action = None

    # ------------------------------------------------------------- setup

    def create(self):
        import genesis as gs
        self.entity = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_path,
                fixed=True,
                requires_jac_and_IK=True,
                decimate=False,
                convexify=True,
                decompose_robot_error_threshold=0.0,
                coacd_options=gs.options.CoacdOptions(
                    threshold=0.05,
                    max_convex_hull=-1,
                    preprocess_resolution=40,
                    resolution=1500,
                    mcts_iterations=120,
                    mcts_max_depth=4,
                ),
                prioritize_urdf_material=True,
            ),
        )
        return self.entity

    def setup(self):
        '''Resolve joint indices and set control gains (call after scene.build()).'''

        def dof_of(name):
            idx = self.entity.get_joint(name).dofs_idx_local
            return int(np.asarray(idx).reshape(-1)[0])

        self.arm_dofs = [dof_of(n) for n in ARM_JOINT_NAMES]
        self.gripper_dofs = [dof_of(n) for n in GRIPPER_JOINT_NAMES]
        self.all_dofs = self.arm_dofs + self.gripper_dofs
        # The current simulation URDF pins two arm joints to zero range
        # (twist, lower-arm fixed); IK uses the remaining 5 movable joints.
        self.movable_arm_dofs = [self.arm_dofs[i] for i in (0, 1, 3, 5, 6)]
        self.ee_link = self.entity.get_link(EE_LINK_NAME)

        # Stiffer than the reference gains (800/80): with collision meshes and
        # inertias parsed correctly, those left ~3 cm of gravity sag at the EE.
        self.entity.set_dofs_kp(
            np.concatenate([ARM_KP, [GRIPPER_KP, GRIPPER_KP]]),
            self.all_dofs,
        )
        self.entity.set_dofs_kv(
            np.concatenate([ARM_KV, [GRIPPER_KV, GRIPPER_KV]]),
            self.all_dofs,
        )
        self.entity.set_dofs_force_range(
            np.array(
                [-87, -87, -87, -87, -12, -12, -12,
                 -GRIPPER_FORCE_LIMIT, -GRIPPER_FORCE_LIMIT],
                dtype=np.float64,
            ),
            np.array(
                [87, 87, 87, 87, 12, 12, 12,
                 GRIPPER_FORCE_LIMIT, GRIPPER_FORCE_LIMIT],
                dtype=np.float64,
            ),
            self.all_dofs,
        )
        lower, upper = self.entity.get_dofs_limit(self.all_dofs)
        self.dof_lower = self._np(lower).astype(np.float64)
        self.dof_upper = self._np(upper).astype(np.float64)
        self.command_lower = self.dof_lower.copy()
        self.command_upper = self.dof_upper.copy()
        self.command_lower[7:] = np.maximum(self.command_lower[7:], GRIPPER_CLOSE)

    # ------------------------------------------------------------ control

    def reset(self):
        self.entity.set_dofs_position(INIT_QPOS, self.all_dofs, zero_velocity=True)
        self.entity.control_dofs_position(INIT_QPOS, self.all_dofs)
        self.last_ik_action = None

    def control_qpos(self, qpos_target):
        '''Command absolute joint positions (9-dim, clipped to task limits).'''
        qpos_target = np.clip(np.asarray(qpos_target, dtype=np.float64),
                              self.command_lower, self.command_upper)
        self.entity.control_dofs_position(qpos_target, self.all_dofs)

    def get_qpos(self):
        return self._np(self.entity.get_dofs_position(self.all_dofs)).astype(np.float64)

    def get_ee_pos(self):
        return self._np(self.ee_link.get_pos()).astype(np.float64)

    def solve_ik(self, target_pos, gripper):
        '''Return the 8-dim joint action [7 arm, 1 gripper] that moves the EE
        toward `target_pos` with a top-down approach axis (yaw left free).

        Restricting IK to the 5 movable joints and aligning only the approach
        axis replaces the reference's post-hoc clipping of the IK solution,
        which corrupted poses whenever unrestricted IK moved the zero-range
        joints (position errors of tens of centimeters).
        '''
        target_pos = self.workspace.clip(target_pos)
        current_qpos = self.get_qpos()
        if self.last_ik_action is None:
            prev_action = np.concatenate([current_qpos[:7], [current_qpos[7]]])
        else:
            prev_action = self.last_ik_action.copy()

        init_qpos = current_qpos.copy()
        init_qpos[:7] = prev_action[:7]
        init_qpos[7:] = prev_action[7]
        qpos = self._np(self.entity.inverse_kinematics(
            link=self.ee_link, pos=target_pos, quat=EE_DOWN_QUAT,
            init_qpos=init_qpos,
            dofs_idx_local=self.movable_arm_dofs,
            rot_mask=[False, False, True],  # align the EE z axis only
        )).astype(np.float64)
        raw_action = np.empty(8, dtype=np.float64)
        raw_action[:7] = qpos[self.arm_dofs]
        raw_action[7] = float(gripper)

        max_step = np.array([MAX_IK_JOINT_STEP] * 7 + [MAX_GRIPPER_JOINT_STEP], dtype=np.float64)
        action = prev_action + np.clip(raw_action - prev_action, -max_step, max_step)
        action_lower = np.concatenate([self.command_lower[:7], [self.command_lower[7]]])
        action_upper = np.concatenate([self.command_upper[:7], [self.command_upper[7]]])
        action = np.clip(action, action_lower, action_upper)
        self.last_ik_action = action.copy()
        return action.astype(np.float32)

    @staticmethod
    def _np(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        return np.asarray(x)
