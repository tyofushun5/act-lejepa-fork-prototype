from transformers import PretrainedConfig
from datasets import concatenate_datasets
from tqdm import tqdm

from robo_utils.common import update_episode_indices
from robo_utils.gym_wrapper import NumpyToTorch
from robo_utils.agent_evaluator import AgentEvaluator
import gymnasium as gym

class AgentEvaluatorMultiEnv:
    '''Evaluate a policy across multiple environments and aggregate results.'''
    def __init__(self, policy, config: PretrainedConfig):
        self.config = config
        self.env_names = config.env.env_names
        self.policy = policy
        self.last_env_datasets = {}

    def __call__(self):
        '''
        Rollout and evaluate policy in all configured environments, aggregate results.
        Returns:
            tuple: Aggregated datasets and info dictionary.
        '''
        all_datasets = []
        all_infos = {}
        self.last_env_datasets = {}
        add_key_prefix = lambda d, prefix: {f'{prefix}/{k}': v for k, v in d.items()}

        pbar = tqdm(self.env_names, desc="Evaluating environments")
        for env_name in pbar:
            pbar.set_description(f"Evaluating: {env_name}")
            dataset, info = self._evaluate_env(env_name)
            all_datasets.append(dataset)
            self.last_env_datasets[env_name] = dataset
            all_infos |= add_key_prefix(info, f'Rollout - per environment/{env_name}')
        
        # Aggregate results of multiple envs
        all_datasets = update_episode_indices(concatenate_datasets(all_datasets))
        group_info = AgentEvaluator.get_rollout_info(all_datasets)
        group_info = add_key_prefix(group_info, 'Rollout')
        all_infos |= group_info
        return all_datasets, all_infos

    def _evaluate_env(self, env_name: str):
        '''
        Evaluate a single environment and return the evaluation dataset and info.
        Note that environments are created from scratch each time to ensure reproducible
        runs, as some environments (Robocasa) do not have a deterministic reset.
        '''
        env = self._create_env(env_name)
        evaluator = AgentEvaluator(
            self.policy,
            env,
            seed=self.config.env.seed,
            num_envs=self.config.env.num_episodes
        )
        dataset, info = evaluator()
        env.close()
        return dataset, info

    def _create_env(self, env_name: str):
        '''Create and return a new environment instance.'''
        env = gym.make(env_name=env_name, **self.config.env['env_kwargs'])
        env = NumpyToTorch(env, device=self.policy.device)
        return env
