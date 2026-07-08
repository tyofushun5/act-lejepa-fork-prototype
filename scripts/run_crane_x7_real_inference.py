#!/usr/bin/env python3
"""Run a trained CRANE-X7 policy against ROS 2 real-robot topics.

This node is intentionally dry-run by default. Add ``--execute`` only after
checking the logged actions with the real CRANE-X7 safely powered and observed.

Typical setup:

  ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
  python -m scripts.run_crane_x7_real_inference \
      --config_path configs/crane_x7/act-jepa.yaml

The policy action convention matches ``custom_envs/crane_x7_env``:
7 absolute arm joint positions plus one gripper position.
"""

from init import init; init()

import argparse
import os
from pathlib import Path
from typing import Optional

if not os.environ.get('MPLCONFIGDIR'):
    mpl_config_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'act-jepa-matplotlib'
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_config_dir)
if not os.environ.get('ROS_LOG_DIR'):
    ros_log_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'act-jepa-ros-logs'
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_LOG_DIR'] = str(ros_log_dir)

import numpy as np
import torch
from safetensors.torch import load_file

from configs.training import Config
from robo_utils.train_utils import (
    DefaultProcessor,
    Metadata,
    get_best_model_checkpoint,
    get_policy,
)


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

# URDF limits used by the CRANE-X7 description currently vendored in this repo.
# The gripper lower bound is intentionally 0.0 to match the training command
# convention: close at contact, do not command the mechanical squeeze limit.
ARM_LOWER = np.array([
    -2.7401669256310974,
    -1.5707963267948966,
    0.0,
    -2.775073510670984,
    0.0,
    -1.5707963267948966,
    -2.91469985083053,
], dtype=np.float64)
ARM_UPPER = np.array([
    2.7401669256310974,
    1.5707963267948966,
    0.0,
    1.7453292519943296e-05,
    0.0,
    1.5707963267948966,
    2.91469985083053,
], dtype=np.float64)


def _load_policy(config_path: str, checkpoint_path: Optional[str], device: torch.device):
    config = Config.load(config_path)
    metadata = Metadata.from_hf(
        config.dataset.repo_ids[0],
        config.dataset.get('revision', 'main'),
    )
    processor = DefaultProcessor(config, metadata)
    policy = get_policy(config, metadata, processor)
    try:
        checkpoint = Path(checkpoint_path) if checkpoint_path else Path(get_best_model_checkpoint(config))
    except FileNotFoundError as exc:
        raise SystemExit(
            f'Could not find trainer state for {config.app!r}: {exc}\n'
            'Pass --checkpoint_path /path/to/model.safetensors explicitly.'
        ) from exc
    if not checkpoint.is_file():
        candidates = sorted(Path('logs/training').glob('**/model.safetensors'))
        message = [
            f'Checkpoint not found: {checkpoint}',
            'Replace checkpoint-XXXX with a real checkpoint directory, or pass an absolute path.',
        ]
        if candidates:
            message.append('Available local checkpoints:')
            message.extend(f'  {path}' for path in candidates[-10:])
        else:
            message.append('No local checkpoints found under logs/training.')
        raise SystemExit('\n'.join(message))
    policy.load_state_dict(load_file(checkpoint, device='cpu'))
    policy.to(device)
    policy.eval()
    return policy, config, checkpoint


def _image_to_rgb_array(msg):
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {'rgb8', 'bgr8'}:
        channels = 3
        row = data.reshape(msg.height, msg.step)
        image = row[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        if encoding == 'bgr8':
            image = image[..., ::-1]
        return np.ascontiguousarray(image)

    if encoding in {'rgba8', 'bgra8'}:
        channels = 4
        row = data.reshape(msg.height, msg.step)
        image = row[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        if encoding == 'bgra8':
            image = image[..., [2, 1, 0, 3]]
        return np.ascontiguousarray(image[..., :3])

    if encoding == 'mono8':
        row = data.reshape(msg.height, msg.step)
        image = row[:, :msg.width].reshape(msg.height, msg.width)
        return np.repeat(image[..., None], 3, axis=-1)

    raise ValueError(f'unsupported image encoding: {msg.encoding!r}')


def _latest_positions_by_name(joint_state_msg):
    return {
        name: float(position)
        for name, position in zip(joint_state_msg.name, joint_state_msg.position)
    }


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Run ACT/ACT-JEPA CRANE-X7 inference on ROS 2 real-robot topics.',
    )
    parser.add_argument('--config_path', default='configs/crane_x7/act-jepa.yaml')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--joint_states_topic', default='/joint_states')
    parser.add_argument('--image_topic', default='/camera/color/image_raw')
    parser.add_argument('--arm_command_topic', default='/crane_x7_arm_controller/joint_trajectory')
    parser.add_argument('--gripper_action', default='/crane_x7_gripper_controller/gripper_cmd')
    parser.add_argument('--rate_hz', type=float, default=10.0)
    parser.add_argument('--arm_duration', type=float, default=0.2)
    parser.add_argument('--max_arm_delta', type=float, default=0.10)
    parser.add_argument('--max_gripper_delta', type=float, default=0.05)
    parser.add_argument('--gripper_lower', type=float, default=0.0)
    parser.add_argument('--gripper_upper', type=float, default=0.9)
    parser.add_argument('--gripper_max_effort', type=float, default=0.0)
    parser.add_argument('--gripper_min_interval', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--stale_timeout', type=float, default=1.0)
    parser.add_argument('--no_gripper', action='store_true')
    parser.add_argument('--execute', action='store_true',
                        help='send commands to the CRANE-X7 controllers')
    parsed, ros_args = parser.parse_known_args(args)

    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node

    from builtin_interfaces.msg import Duration as DurationMsg
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
    from sensor_msgs.msg import Image, JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class CraneX7PolicyNode(Node):
        def __init__(self):
            super().__init__('act_jepa_crane_x7_real_inference')
            if parsed.device == 'auto':
                device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device_name = parsed.device
            self.device = torch.device(device_name)
            self.policy, self.config, self.checkpoint = _load_policy(
                parsed.config_path,
                parsed.checkpoint_path,
                self.device,
            )
            self.policy.reset()

            self.latest_joint_state = None
            self.latest_joint_stamp = None
            self.latest_image = None
            self.latest_image_stamp = None
            self.step_count = 0
            self.last_gripper_goal = None
            self.last_gripper_send_time = None
            self.gripper_goal_future = None

            self.create_subscription(
                JointState,
                parsed.joint_states_topic,
                self._on_joint_state,
                10,
            )
            self.create_subscription(
                Image,
                parsed.image_topic,
                self._on_image,
                10,
            )
            self.arm_publisher = self.create_publisher(
                JointTrajectory,
                parsed.arm_command_topic,
                10,
            )
            self.gripper_client = None
            if not parsed.no_gripper:
                self.gripper_client = ActionClient(
                    self,
                    GripperCommand,
                    parsed.gripper_action,
                )

            period = 1.0 / parsed.rate_hz
            self.timer = self.create_timer(period, self._tick)
            mode = 'EXECUTE' if parsed.execute else 'DRY-RUN'
            self.get_logger().info(
                f'{mode}: config={parsed.config_path} checkpoint={self.checkpoint} '
                f'device={self.device} rate={parsed.rate_hz}Hz'
            )
            if not parsed.execute:
                self.get_logger().warn('Dry-run mode: actions are logged but not sent.')

        def _on_joint_state(self, msg):
            self.latest_joint_state = msg
            self.latest_joint_stamp = self.get_clock().now()

        def _on_image(self, msg):
            try:
                self.latest_image = _image_to_rgb_array(msg)
                self.latest_image_stamp = self.get_clock().now()
            except ValueError as exc:
                self.get_logger().warn(str(exc), throttle_duration_sec=2.0)

        def _tick(self):
            if parsed.max_steps > 0 and self.step_count >= parsed.max_steps:
                self.get_logger().info('Reached max_steps; stopping inference timer.')
                self.timer.cancel()
                return

            if not self._inputs_ready():
                return

            state = self._make_state_vector()
            if state is None:
                return

            action = self._infer_action(state, self.latest_image)
            arm_target, gripper_target = self._make_safe_targets(action, state)
            self.step_count += 1

            if parsed.execute:
                self._publish_arm_target(arm_target)
                if not parsed.no_gripper:
                    self._send_gripper_target(gripper_target)
            elif self.step_count == 1 or self.step_count % max(1, int(parsed.rate_hz)) == 0:
                self.get_logger().info(
                    f'dry-run step={self.step_count} '
                    f'arm={np.round(arm_target, 3).tolist()} '
                    f'gripper={gripper_target:.3f}'
                )

        def _inputs_ready(self):
            if self.latest_joint_state is None or self.latest_image is None:
                self.get_logger().info(
                    'Waiting for joint state and image messages...',
                    throttle_duration_sec=2.0,
                )
                return False

            now = self.get_clock().now()
            timeout = Duration(seconds=parsed.stale_timeout)
            if now - self.latest_joint_stamp > timeout:
                self.get_logger().warn('Latest joint state is stale.', throttle_duration_sec=2.0)
                return False
            if now - self.latest_image_stamp > timeout:
                self.get_logger().warn('Latest image is stale.', throttle_duration_sec=2.0)
                return False
            return True

        def _make_state_vector(self):
            positions = _latest_positions_by_name(self.latest_joint_state)
            missing = [name for name in ARM_JOINT_NAMES if name not in positions]
            if missing:
                self.get_logger().warn(
                    f'Missing arm joints in {parsed.joint_states_topic}: {missing}',
                    throttle_duration_sec=2.0,
                )
                return None

            arm = [positions[name] for name in ARM_JOINT_NAMES]
            finger_a = positions.get(GRIPPER_JOINT_NAMES[0])
            finger_b = positions.get(GRIPPER_JOINT_NAMES[1], finger_a)
            if finger_a is None:
                self.get_logger().warn(
                    f'Missing gripper joint {GRIPPER_JOINT_NAMES[0]!r}.',
                    throttle_duration_sec=2.0,
                )
                return None
            return np.asarray(arm + [finger_a, finger_b], dtype=np.float32)

        def _infer_action(self, state, image):
            obs = {
                'observation.state': torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ),
                'observation.image': torch.as_tensor(
                    image,
                    dtype=torch.uint8,
                    device=self.device,
                ),
                'task_index': torch.as_tensor(
                    parsed.task_index,
                    dtype=torch.long,
                    device=self.device,
                ),
            }
            action = self.policy.select_action([obs])
            return action.squeeze(0).detach().cpu().numpy().astype(np.float64)

        def _make_safe_targets(self, action, state):
            raw_arm = np.clip(action[:7], ARM_LOWER, ARM_UPPER)
            current_arm = state[:7].astype(np.float64)
            delta = np.clip(
                raw_arm - current_arm,
                -parsed.max_arm_delta,
                parsed.max_arm_delta,
            )
            arm_target = np.clip(current_arm + delta, ARM_LOWER, ARM_UPPER)

            raw_gripper = float(np.clip(action[7], parsed.gripper_lower, parsed.gripper_upper))
            current_gripper = float(state[7])
            gripper_delta = np.clip(
                raw_gripper - current_gripper,
                -parsed.max_gripper_delta,
                parsed.max_gripper_delta,
            )
            gripper_target = float(
                np.clip(current_gripper + gripper_delta, parsed.gripper_lower, parsed.gripper_upper)
            )
            return arm_target, gripper_target

        def _publish_arm_target(self, arm_target):
            msg = JointTrajectory()
            msg.joint_names = list(ARM_JOINT_NAMES)
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in arm_target]
            duration = Duration(seconds=parsed.arm_duration).to_msg()
            point.time_from_start = DurationMsg(sec=duration.sec, nanosec=duration.nanosec)
            msg.points.append(point)
            self.arm_publisher.publish(msg)

        def _send_gripper_target(self, gripper_target):
            if self.gripper_client is None:
                return
            now = self.get_clock().now()
            if self.last_gripper_send_time is not None:
                if (now - self.last_gripper_send_time).nanoseconds < parsed.gripper_min_interval * 1e9:
                    return
            if self.gripper_goal_future is not None and not self.gripper_goal_future.done():
                return
            if self.last_gripper_goal is not None:
                if abs(gripper_target - self.last_gripper_goal) < 1e-3:
                    return
            if not self.gripper_client.server_is_ready():
                self.get_logger().warn(
                    f'Gripper action server not ready: {parsed.gripper_action}',
                    throttle_duration_sec=2.0,
                )
                return

            goal = GripperCommand.Goal()
            goal.command.position = float(gripper_target)
            goal.command.max_effort = float(parsed.gripper_max_effort)
            self.gripper_goal_future = self.gripper_client.send_goal_async(goal)
            self.last_gripper_send_time = now
            self.last_gripper_goal = float(gripper_target)

    rclpy.init(args=ros_args)
    node = CraneX7PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
