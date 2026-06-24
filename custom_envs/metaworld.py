import gymnasium as gym
import numpy as np
from metaworld.sawyer_xyz_env import SawyerXYZEnv
from functools import cached_property

class FixedResetWrapper(gym.Wrapper):
    '''
    Gymnasium wrapper for MetaWorld Sawyer environments with improved reset behavior.
    Fixes the issue where Metaworld completely ignores the seed in reset(). 
    In contrast, this class ensures proper randomization of goal and hand positions.
    Importantly, each seed is deterministic, and different seeds, create different goal and hand positions.
    '''
    def __init__(self, env: SawyerXYZEnv, is_random_hand_pos=True):
        super().__init__(env)
        self.is_random_hand_pos = is_random_hand_pos
        # override defaults so we can randomize goal pos
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped._set_task_called = True
        # Store original hand position
        self._original_hand_init_pos = np.copy(self.unwrapped.hand_init_pos)

    def randomize_hand_position(self):
        '''Randomize the robot hand initial position within allowed bounds.'''
        # Restore hand position before randomizing for deterministic results
        default_pos = self._original_hand_init_pos  
        # Add random noise to each coordinate
        noise = np.random.uniform(-0.05, 0.05, size=default_pos.shape)
        random_hand_pos = default_pos + noise
        # Clip to valid bounds
        random_hand_pos = np.clip(random_hand_pos, self.unwrapped.hand_low, self.unwrapped.hand_high)
        # Overwrite hand position - this is used during super().reset()
        self.unwrapped.hand_init_pos = random_hand_pos

    def reset(self, *, seed = None, options = None):
        '''
        Fixes strange reset behavior with Metaworld.
        How it's done:
        - set `_freeze_rand_vec` to False (before),
        - overwrite reset method
            1. seed with numpy
            2. randomize hand position
            3. reset environment - this will randomize goal position
        '''
        np.random.seed(seed)
        if self.is_random_hand_pos:
            self.randomize_hand_position()
        obs, info = super().reset()
        return obs, info


class FixedStepWrapper(gym.Wrapper):
    '''
    Gym wrapper that clips actions to the range [-1, 1] before passing them to the environment.
    Ensures all actions remain within valid bounds, preventing unexpected behavior.
    '''
    def step(self, action):
        # Make sure that action is in range -1, 1
        action = action.clip(self.env.action_space.low, self.env.action_space.high)
        return super().step(action)    


class FullyObservableWrapper(gym.Wrapper):
    '''
    Make the environment fully observable, ensuring the goal position is included in observations.
    Important when goal visibility is required (e.g., scripted policies).
    '''
    def __init__(self, env):
        super().__init__(env)
        self._make_fully_observable()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._make_fully_observable()
        return obs, info

    def _make_fully_observable(self):
        '''
        To make it observable, '_partially_observable' is set to False. Also, cached 
        'sawyer_observation_space' property is deleted, to force recomputation, as it depends 
        on _partially_observable. With these changes, goal position information will be available in 
        the observation (otherwise it's clipped to 0 in .step() and .reset() methods).
        '''
        self.unwrapped._partially_observable = False
        if hasattr(self.unwrapped, 'sawyer_observation_space'):
            del self.unwrapped.sawyer_observation_space
        
        # trigger recompute and update observation_space
        self.unwrapped.observation_space = self.unwrapped.sawyer_observation_space


class ObsWrapper(gym.Wrapper):
    '''
    Gym wrapper for Metaworld environments to modify observation space and camera settings.
    Adds image and state observations, and configures camera for better visualization.
    '''
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # zoom in a bit on cam 2
        env.unwrapped.model.cam_pos[2] = [0.75, 0.075, 0.7]
        
        img_size = env.unwrapped.mujoco_renderer.width
        self._setup_obs_space()
        self._setup_default_cam(img_size)

    def _setup_obs_space(self):
        '''
        Define the observation space for the environment wrapper.
        Sets up image keys, state key, task_index, etc.
        '''
        img_size = self.unwrapped.mujoco_renderer.width
        self.observation_space = gym.spaces.Dict({
            'observation.image': gym.spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
            'observation.environment_state': self.observation_space, # Box (39,)
            'observation.state': gym.spaces.Box(self.observation_space.low[:4], self.observation_space.high[:4], (4,), float),
            'task_index': gym.spaces.Box(0, np.inf, (), int),
        })

    def _setup_default_cam(self, img_size):
        '''
        Set up the default camera configuration for rendering.
        Args:
            img_size (int): Size of the rendered image.
        '''
        default_cam_config = {
            "distance": 1.5,
            "azimuth": 20,
            "elevation": -30.0,
            "lookat": np.array([0, 0.5, 0.1]),
        }

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self.unwrapped.mujoco_renderer = MujocoRenderer(
            self.unwrapped.model,
            self.unwrapped.data,
            default_cam_config,
            camera_id=-1,
            width=img_size,
            height=img_size
        )

    def reset(self, *, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._prepare_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = np.float32(reward) # Ensure reward is float32
        obs = self._prepare_obs(obs)
        return obs, reward, terminated, truncated, info

    def _prepare_obs(self, obs):
        '''
        Prepare observation dict with both full environment state and agent state.
        The observation includes:
        - 'observation.state': 3D end-effector position & gripper openness (first 4 elements).
        - 'observation.environment_state': full environment state
        - 'observation.image': rendered image
        
        More info at: https://metaworld.farama.org/benchmark/state_space/
        '''
        return {
            'observation.state': obs[:4],
            'observation.environment_state': obs,
            'observation.image': self.render(),
            'task_index': self.task_index
        }

    def render(self, camera_id=-1):
        '''Render the environment from the specified camera.'''
        self.unwrapped.mujoco_renderer.camera_id = camera_id
        return super().render()

    @cached_property
    def task_index(self):
        env_name = self.spec.kwargs['env_name']
        
        from custom_envs.env_config import MetaworldEnvConfig
        tasks = MetaworldEnvConfig.env_tasks()
        id = tasks.set_index('env_name').loc[env_name, 'id']
        # Gym env checker expects id to be np.array not np.int
        # This is important for vec envs to properly stack ids
        return np.asarray(id)


class TerminateOnSuccessWrapper(gym.Wrapper):
    '''
    Wrapper that terminates the episode when the 'success' flag is set in the info dict.
    '''

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = info["success"] != 0
        return obs, reward, terminated, truncated, info


def _setup_rendering():
    '''
    Automatically set the MUJOCO_GL environment variable for
    MetaWorld env based on the OS.
    '''
    import platform
    import os
    if platform.system() == 'Darwin':
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        os.environ['MUJOCO_GL'] = 'egl'


def make_metaworld_env(
    env_name: str, 
    render_mode='rgb_array', 
    img_size=128,
    is_random_hand_pos=True,
    terminate_on_success=True,
):
    '''
    Factory function to create a Metaworld environment with wrappers.
    Args:
        env_name (str): Name of the Metaworld environment.E.g., `button-press-v3`
        render_mode (str): Rendering mode.
        img_size (int): Image width and height.
        is_random_hand_pos (bool): Randomize hand position on reset.
        terminate_on_success (bool): Terminate episode on success.
        max_episode_steps (int): Maximum steps per episode.
    Returns:
        gym.Env: Wrapped Metaworld environment.
    '''
    _setup_rendering()

    from metaworld.env_dict import ALL_V3_ENVIRONMENTS
    env_cls = ALL_V3_ENVIRONMENTS[env_name]

    env = env_cls(render_mode=render_mode, width=img_size, height=img_size)
    env = FixedResetWrapper(env, is_random_hand_pos)
    env = FixedStepWrapper(env)
    env = FullyObservableWrapper(env)
    env = ObsWrapper(env)
    env = TerminateOnSuccessWrapper(env) if terminate_on_success else env
    return env


gym.register(id='Metaworld-v3', entry_point=lambda **kwargs: make_metaworld_env(**kwargs))
