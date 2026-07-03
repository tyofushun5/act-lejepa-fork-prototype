from transformers import PretrainedConfig
from datasets import concatenate_datasets
from tqdm import tqdm
from copy import deepcopy

from robo_utils.common import update_episode_indices
from robo_utils.gym_wrapper import NumpyToTorch
from robo_utils.agent_evaluator import AgentEvaluator
import gymnasium as gym

class AgentEvaluatorMultiEnv:
    '''Evaluate a policy across multiple environments and aggregate results.'''
    def __init__(self, policy, config: PretrainedConfig):
        self.config = config
        self.env_names = config.env.env_names
        self.env_specs = self._make_env_specs()
        self.policy = policy
        self.last_env_datasets = {}
        self.last_env_infos = {}

    def __call__(self):
        '''
        Rollout and evaluate policy in all configured environments, aggregate results.
        Returns:
            tuple: Aggregated datasets and info dictionary.
        '''
        all_datasets = []
        all_infos = {}
        self.last_env_datasets = {}
        self.last_env_infos = {}
        add_key_prefix = lambda d, prefix: {f'{prefix}/{k}': v for k, v in d.items()}

        pbar = tqdm(self.env_specs, desc="Evaluating environments")
        for label, env_name, env_kwargs in pbar:
            pbar.set_description(f"Evaluating: {label}")
            dataset, info = self._evaluate_env(env_name, env_kwargs)
            all_datasets.append(dataset)
            self.last_env_datasets[label] = dataset
            self.last_env_infos[label] = info
            all_infos |= add_key_prefix(info, f'Rollout - per environment/{label}')
        
        # Aggregate results of multiple envs
        all_datasets = update_episode_indices(concatenate_datasets(all_datasets))
        group_info = AgentEvaluator.get_rollout_info(all_datasets)
        group_info = add_key_prefix(group_info, 'Rollout')
        all_infos |= group_info
        return all_datasets, all_infos

    def _make_env_specs(self):
        '''Build concrete evaluation environment variants.'''
        env_kwargs = dict(self.config.env['env_kwargs'])
        camera_views = self.config.env.get('eval_camera_views')
        specs = []

        for env_name in self.env_names:
            if not camera_views:
                specs.append((env_name, env_name, deepcopy(env_kwargs)))
                continue

            for camera_view in camera_views:
                view_kwargs = deepcopy(env_kwargs)
                view_kwargs['camera_view'] = camera_view
                specs.append((f'{env_name}/{camera_view}', env_name, view_kwargs))

        return specs

    def get_camera_view_infos(self):
        '''Return the latest per-camera rollout metrics keyed by camera view.'''
        camera_views = [str(view) for view in self.config.env.get('eval_camera_views', [])]
        if len(camera_views) < 2:
            return {}

        infos = {}
        for label, info in self.last_env_infos.items():
            camera_view = str(label).rsplit('/', 1)[-1]
            if camera_view in camera_views:
                infos[camera_view] = info
        return infos

    def _evaluate_env(self, env_name: str, env_kwargs: dict):
        '''
        Evaluate a single environment and return the evaluation dataset and info.
        Note that environments are created from scratch each time to ensure reproducible
        runs, as some environments (Robocasa) do not have a deterministic reset.
        '''
        env = self._create_env(env_name, env_kwargs)
        evaluator = AgentEvaluator(
            self.policy,
            env,
            seed=self.config.env.seed,
            num_envs=self.config.env.num_episodes
        )
        dataset, info = evaluator()
        env.close()
        return dataset, info

    def _create_env(self, env_name: str, env_kwargs: dict):
        '''Create and return a new environment instance.'''
        env = gym.make(env_name=env_name, **env_kwargs)
        env = NumpyToTorch(env, device=self.policy.device)
        return env
