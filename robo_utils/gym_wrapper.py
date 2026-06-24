# Define wrapper for the environment
import gymnasium as gym
import numpy as np
import cv2
from numbers import Number

class GymWrapper(gym.Wrapper):
    def __init__(self, env, img_size=128):
        '''
        Args:
            env (gym.Env): The environment to wrap.
            img_size (int): The size to which the environment image will be resized.
        '''
        super().__init__(env)
        assert env.render_mode is not None, '`render_mode` must be set.'
        self.img_size = img_size
        
        # Update observation space to include state and image
        state_space = self.observation_space
        image_space = gym.spaces.Box(0, 255, (self.img_size, self.img_size, 3), np.uint8)
        obs_space = {
            'observation.state': state_space,
            'observation.image': image_space
        }
        self.observation_space = gym.spaces.Dict(obs_space)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._prepare_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._prepare_obs(obs)
        return obs, info

    def _prepare_obs(self, obs):
        '''
        Prepare the observation by separating it into state and image components.
        
        Args:
            obs: The observation to be prepared.
        
        Returns:
            dict: A dictionary containing the state and image from the observation.
        '''
        x = {'observation.state': obs, 'observation.image': self.render()}
        return x

    def render(self):
        img = super().render()
        resized_img = cv2.resize(img, (self.img_size, self.img_size))
        return resized_img


import gymnasium as gym
import torch

class _NumpyToTorchMixin:
    '''
    Converts all numpy arrays in dict observations to torch tensors.
    
    Expects observations like:
        {
            'observation.state': np.ndarray,
            'observation.image': np.ndarray,
            ...
        }
    All numpy arrays are converted to torch tensors on the given device.
    '''
    def __init__(self, env: gym.Env, device='cpu', default_dtype=torch.float32):
        '''
        Args:
            env (gym.Env): The environment to wrap. Must have Dict observation space.
            device (str or torch.device): Device to place tensors on.
            default_dtype (torch.dtype): Default dtype for floats arrays.
        '''
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.default_dtype = default_dtype
        self.device = device
    
    def _np_to_torch(self, x: dict) -> dict:
        '''
        Convert numpy arrays and scalars in a dict to torch tensors.
        Handles negative strides for numpy arrays by copying them.
        Non-numeric types (e.g., strings) are left unchanged. 
        
        Args:
            x (dict): Input dictionary with observation values.
        Returns:
            dict: Dictionary with numpy arrays and scalars converted to torch tensors.
        '''
        for k, v in x.items():
            if isinstance(v, (np.ndarray, Number)):
                dtype = self.default_dtype if 'float' in str(v.dtype) else None
                # Ensure positive strides for torch compatibility
                if v.strides and any(s < 0 for s in v.strides):
                    v = v.copy()
                x[k] = torch.as_tensor(v, dtype=dtype, device=self.device)
        return x

class _NumpyToTorchSingle(_NumpyToTorchMixin, gym.ObservationWrapper):
    def observation(self, obs):
        return self._np_to_torch(obs)
    
class _NumpyToTorchVec(_NumpyToTorchMixin, gym.vector.VectorObservationWrapper):
    def observations(self, obs):
        return self._np_to_torch(obs)

def NumpyToTorch(env, device='cpu', default_dtype=torch.float32):
    '''
    Converts all numpy arrays in dict observations to torch tensors.
    
    Expects observations like:
        {
            'observation.state': np.ndarray,
            'observation.image': np.ndarray,
            ...
        }
    All numpy arrays are converted to torch tensors on the given device.
    '''
    if isinstance(env, gym.vector.VectorEnv):
        return _NumpyToTorchVec(env, device, default_dtype)
    else:
        return _NumpyToTorchSingle(env, device, default_dtype)


if __name__ == '__main__':
    env_wrapper = gym.make_vec(id='MountainCar-v0', render_mode='rgb_array',  num_envs=1, wrappers=(GymWrapper,))
