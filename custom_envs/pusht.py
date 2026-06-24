import gymnasium as gym
import numpy as np

class ClipActionWrapper(gym.Wrapper):
    '''
    Gym wrapper that clips actions to the valid range [-1, 1] before passing them to the environment.
    Ensures all actions remain within valid bounds, preventing unexpected behavior.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.low = np.asarray(self.env.action_space.low)
        self.high = np.asarray(self.env.action_space.high)

    def step(self, action):
        # Make sure that action is in valid range
        action = action.clip(self.low, self.high)
        return super().step(action)


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, img_size: int):
        super().__init__(env)
        self._setup_obs_space(img_size)
    
    def _setup_obs_space(self, img_size):
        self.env.unwrapped.observation_width = img_size
        self.env.unwrapped.observation_height = img_size
        self.env.unwrapped.visualization_width = img_size
        self.env.unwrapped.visualization_height = img_size
        
        self.observation_space = gym.spaces.Dict({
            'observation.state': self.env.observation_space['agent_pos'],
            'observation.environment_state': self.env.observation_space['environment_state'],
            'observation.image' : gym.spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
            'task_index': gym.spaces.Box(0, np.inf, (), int)
        })

    def observation(self, obs: dict):
        obs_state = obs.pop('agent_pos')
        obs_env_state = obs.pop('environment_state')
        img = self.env.unwrapped._render(visualize=False)
        return {
            'observation.state': obs_state,
            'observation.environment_state': obs_env_state,
            'observation.image': img,
            'task_index': self.task_index
        }
    
    @property
    def task_index(self):
        return np.asarray(0) # There's only one task in this env


def make_pusht_env(
    env_name=None, # leave it empty
    render_mode='rgb_array',
    img_size=128,
):
    import gym_pusht
    env = gym.make(
        id='gym_pusht/PushT-v0',
        obs_type='environment_state_agent_pos',
        render_mode=render_mode,
    )

    # since we use small image size, it really isn't visible 
    # whether 0.95 is covered. We could increase the image size
    # (and collect more data) or simply lower the threshold.
    env.unwrapped.success_threshold = 0.9

    env = ObsWrapper(env, img_size)
    return env

gym.register(
    id='my-pusht-v0',
    entry_point=make_pusht_env
)