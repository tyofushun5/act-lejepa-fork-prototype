import numpy as np
from einops import pack
from functools import cached_property
import gymnasium as gym
import torch
from mani_skill.utils.wrappers import CPUGymWrapper
import mani_skill.envs


class ObsWrapper(gym.ObservationWrapper):
    '''
    Gym wrapper that modifies observation space and camera settings.
    Adds image, state observations, etc.
    '''
    def __init__(self, env, img_size: int):
        super().__init__(env)
        self._setup_obs_space(img_size)
    
    def _setup_obs_space(self, img_size):
        '''
        Define the observation space for the Maniskill environment wrapper.
        Sets up image keys, state key, task index etc.
        '''
        obs_agent_len = sum(v.shape[-1] for k, v in self.env.unwrapped.observation_space['agent'].items()) # 18
        obs_extra_len = sum(v.shape[-1] for k, v in self.env.unwrapped.observation_space['extra'].items()) # 24
        full_obs_len = obs_agent_len + obs_extra_len

        self.observation_space = gym.spaces.Dict({
            'observation.state': gym.spaces.Box(-np.inf, np.inf, (obs_agent_len,), np.float32),
            # 'observation.environment_state': gym.spaces.Box(-np.inf, np.inf, (full_obs_len, ), np.float32),
            'observation.image': gym.spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
            'task_index': gym.spaces.Box(0, np.inf, (), int),
        })
        if self.env.unwrapped.num_envs > 1:
            from gymnasium.vector.utils import batch_space
            self.observation_space = batch_space(self.observation_space, self.env.unwrapped.num_envs)

    def observation(self, obs: dict):
        '''
        Prepare observation dict with agent state and images.
        '''
        # concatenate states
        num_envs = self.unwrapped.num_envs
        pattern = 'b *' if num_envs > 1 else '*'

        # Agent state (proprioceptive state)
        obs_state, _ = pack(
            [obs['agent']['qpos'], obs['agent']['qvel']], 
            pattern)
        
        # # Full observation state
        # obs_env_state, _ = pack(
        #     [v for v in obs['agent'].values()] + [v for v in obs['extra'].values()],
        #     pattern
        # )
        
        # get images
        imgs = self.render()

        # get task index (might be batched)
        task_index = self.task_index.to(device=obs_state.device)
        if num_envs > 1:
            task_index = task_index.repeat(num_envs)

        return {
            'observation.state': obs_state,
            # 'observation.environment_state': obs_env_state,
            'observation.image': imgs,
            'task_index': task_index
        }

    def render(self, camera_name:str='render_camera', obs:dict=None):
        if camera_name == 'render_camera':
            img = super().render()
        else:
            img = obs['sensor_data'][camera_name]['rgb']
        return img

    @cached_property
    def task_index(self):
        env_name = self.spec.kwargs['env_name']
        
        from custom_envs.env_config import ManiSkillEnvConfig
        tasks = ManiSkillEnvConfig.env_tasks()
        id = tasks.set_index('env_name').loc[env_name, 'id']
        # Gym env checker expects id to be np.array not np.int
        # This is important for vec envs to properly stack ids
        return torch.asarray(id)

    @property
    def camera_names(self):
        return list(self.env.unwrapped._sensors.keys()) + list(self.env.unwrapped._human_render_cameras.keys())


class ClipActionWrapper(gym.Wrapper):
    '''
    Gym wrapper that clips actions to the valid range [-1, 1] before passing them to the environment.
    Ensures all actions remain within valid bounds, preventing unexpected behavior.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.low = torch.asarray(self.env.action_space.low)
        self.high = torch.asarray(self.env.action_space.high)

    def step(self, action):
        # Make sure that action is in valid range
        action = action.clip(self.low, self.high)
        return super().step(action)


class FixNumpyWrapper(gym.Wrapper):
    '''
    When simulating on GPU, even terminated and truncated will become 
    torch tensors, which messes up some evaluation code, as that's 
    not a default behavior. To fix it, we simply convert terminated
    and truncated to numpy arrays.
    '''
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = np.asarray(terminated.cpu())
        truncated = np.asarray(truncated.cpu())
        return obs, reward, terminated, truncated, info


def make_mani_skill_env(
    env_name: str,
    render_mode='rgb_array',
    img_size=128,
    obs_mode='state_dict',
    num_envs=1, # trigger gpu rendering with num_envs > 1
    **env_kwargs
):
    '''
    Factory function to create a ManiSkill environment with wrappers.
    '''
    camera_config = dict(
        width=img_size, height=img_size,
        pose=torch.tensor([ 0.5000, -0.5000,  0.8000,  0.3402, -0.2702,  0.1027,  0.8948])
    )

    env = gym.make(
        id=env_name,
        render_mode=render_mode,
        sensor_configs=camera_config,
        human_render_camera_configs=camera_config,
        obs_mode=obs_mode,
        num_envs=num_envs,
        **env_kwargs,
    )
    env = env.unwrapped
    
    env.metadata['render_fps'] = 30
    
    env = ClipActionWrapper(env)
    env = ObsWrapper(env, img_size)

    if num_envs == 1:
        env = CPUGymWrapper(env)
    else:
        env = FixNumpyWrapper(env)

    
    return env

gym.register(
    id='ManiSkill-v3',
    entry_point=make_mani_skill_env,
    disable_env_checker=True,
)


# This is a hack for ManiSkill-v3
# The goal is to fix max_episode_steps with batched environment (num_envs > 1)
from functools import wraps
from mani_skill.utils.registration import TimeLimitWrapper

_orig_gym_make = gym.make

@wraps(_orig_gym_make)
def _make_and_unwrap(id, *args, **kwargs):
    # Call original gym.make
    env = _orig_gym_make(id, *args, **kwargs)

    # If ManiSkill-v3, replace TimeLimit with fixed TimeLimitWrapper
    # to get torch-batched truncated
    if id == 'ManiSkill-v3':
        if isinstance(env, gym.wrappers.TimeLimit):
            env = env.env # remove TimeLimit
            env = TimeLimitWrapper(env, max_episode_steps=kwargs['max_episode_steps'])
            if env.unwrapped.num_envs == 1: 
                env = CPUGymWrapper(env)

    return env

gym.make = _make_and_unwrap
