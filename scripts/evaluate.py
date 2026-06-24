from init import init; init()
import argparse
from pprint import pprint

import torch
from safetensors.torch import load_file

import custom_envs  # register custom Gymnasium environments
from configs.training import Config
from robo_utils.agent_evaluator_multi_env import AgentEvaluatorMultiEnv
from robo_utils.train_utils import DefaultProcessor, Metadata, get_best_model_checkpoint, get_policy


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained policy from a config file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to a config file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model.safetensors')
    args = parser.parse_args()

    config = Config.load(args.config_path)
    metadata = Metadata.from_hf(config.dataset.repo_ids[0], config.dataset.get('revision', 'main'))
    processor = DefaultProcessor(config, metadata)

    policy = get_policy(config, metadata, processor)
    checkpoint_path = args.checkpoint_path or get_best_model_checkpoint(config)
    policy.load_state_dict(load_file(checkpoint_path, device='cpu'))
    policy.to('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval()

    _, info = AgentEvaluatorMultiEnv(policy, config)()
    pprint(info, sort_dicts=False)


if __name__ == '__main__':
    main()
