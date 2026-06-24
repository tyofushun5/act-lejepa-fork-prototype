import torch
import torch.nn as nn
from datasets import Dataset as HFDataset

import gymnasium as gym
from dataclasses import dataclass
from PIL import Image
from time import time
import numpy as np

class AgentEvaluator:
    '''
    Test the agent in the environment for specified number of episodes (num_envs). 
    '''
    def __init__(self, policy: nn.Module, env: gym.Env, seed = 42, num_envs = 1):
        self.policy = policy
        self.env = env
        self.seed = seed
        self.num_envs = num_envs

        self._replay_buffer = []

    def __call__(self):
        '''
        Collects and returns the dataset alongside info stats.
        '''
        start_time = time()
        rollout_data = []
        for episode_index in range(self.num_envs):
            ep_data = self.rollout(self.seed+episode_index, episode_index)
            rollout_data.extend(ep_data)
            
        rollout_dataset = self._create_rollout_dataset(rollout_data)
        rollout_info = self.get_rollout_info(rollout_dataset)

         # wait for GPU kernels to finish to take time()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        rollout_info['time (s)'] = time() - start_time

        return rollout_dataset, rollout_info
    
    @torch.inference_mode()
    def rollout(self, seed: int, episode_index: int = 0):
        '''
        Executes a single rollout of the policy in the environment.
        Roll out the policy over time to see how it performs in the environment. 
        The policy is applied step by step, and the agent interacts with the environment 
        to observe the outcomes (rewards, new states, etc.).

        Returns: a replay buffer for all data.
        '''
        obs, _ = self.env.reset(seed=seed)
        self.policy.reset()
        self.policy.eval()
        self._replay_buffer = []

        fps = self.env.metadata['render_fps']
        frame_index = 0
        done = False

        while not done:
            inputs = dict(obs)
            # Add a fake batch dim, as policies expect batched inputs
            inputs = [inputs]

            # ------------------- Get action -------------------
            action = self.policy.select_action(inputs).cpu().numpy()
            # Remove the batch dim (1, action_dim) -> (action_dim, )
            action = action.squeeze(0)

            # ------------------- Step in the environment -------------------
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            episode_step = {
                'action': action,
                'episode_index': episode_index,
                'frame_index': frame_index,
                'timestamp': frame_index / fps,
                'next.done': done,
                'terminated': terminated,
                'truncated': truncated,
                'next.reward': reward,
            }

            self._add_to_replay_buffer(episode_step, obs)

            obs = next_obs
            frame_index += 1
            del inputs

        return self._replay_buffer
    
    def _add_to_replay_buffer(self, episode_step: dict, obs: dict):
        '''
        Processes the current observation and adds it to the episode step dictionary.
        Converts tensors to numpy arrays or scalars, and image data to PIL Images.
        Appends the updated episode step to the replay buffer.
        '''
        # Add observations to episode_step
        for k, v in obs.items():
            if isinstance(v, torch.Tensor): 
                v = v.cpu().numpy()
                v = v.item() if v.size == 1 else v
            if 'image' in k: 
                v = Image.fromarray(np.asarray(v))
            episode_step[k] = v

        self._replay_buffer.append(episode_step)

    def _create_rollout_dataset(self, rollout_data: list):
        rollout_dataset = HFDataset.from_list(rollout_data)
        self._validate_episode_count(rollout_dataset)
        return rollout_dataset

    def _validate_episode_count(self, rollout_dataset: HFDataset):
        '''Ensure the dataset contains exactly `num_envs` unique episodes.'''
        ids = rollout_dataset.unique('episode_index')
        assert len(ids) == self.num_envs, f'Expected {self.num_envs} episodes, got {len(ids)}'

    @classmethod
    def get_rollout_info(cls, env_dataset: HFDataset):
        '''Calculate stats for rollout in the dataset such as average sum/max of rewards, etc.'''
        df = env_dataset.to_pandas()
        df_groupby_episode = df.groupby('episode_index')
        rewards_per_episode = df_groupby_episode['next.reward']
        # the episode is considered terminated if any step (in that episode) has terminated=True
        termination_per_episode = df_groupby_episode['terminated'].max()
        num_steps_per_episode = df_groupby_episode.size()

        info = {
            'sum_reward': rewards_per_episode.sum().mean(),
            'sum_reward (median)': rewards_per_episode.sum().median(),
            'max_reward': rewards_per_episode.max().mean(),
            'solved %': termination_per_episode.mean() * 100,
            'steps': num_steps_per_episode.mean()
        }
        info = {k: v.item() for k, v in info.items()}
        return info

    
@dataclass
class RandomPolicy:
    env: gym.Env

    def select_action(self, obs):
        # Convert actions to tensor and add fake batch dim -> (1, action_dim)
        action = self.env.action_space.sample()
        action = torch.from_numpy(action).unsqueeze(0)
        return action
    
    def eval(self):
        pass

    def reset(self):
        pass
