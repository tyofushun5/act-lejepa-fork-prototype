from init import init; init()
import argparse
from pathlib import Path
from pprint import pprint

import torch
from safetensors.torch import load_file

import custom_envs  # register custom Gymnasium environments
from configs.training import Config
from robo_utils.agent_evaluator_multi_env import AgentEvaluatorMultiEnv
from robo_utils.common import datasets, save_episode_video
from robo_utils.train_utils import DefaultProcessor, Metadata, get_best_model_checkpoint, get_policy


def _add_key_prefix(values, prefix):
    if not prefix:
        return values
    return {f'{prefix}/{key}': value for key, value in values.items()}


def _get_wandb_videos(evaluator, dataset, video_prefix):
    env_datasets = getattr(evaluator, 'last_env_datasets', None) or {'rollout': dataset}
    videos = {}

    for env_name, env_dataset in env_datasets.items():
        image_col = next((name for name, feature in env_dataset.features.items()
                          if isinstance(feature, datasets.features.Image)), None)
        if image_col is None:
            continue

        safe_name = str(env_name).replace('/', '_')
        video_dir = Path('logs/wandb/eval') / video_prefix.replace('/', '_') / safe_name
        video_path = save_episode_video(video_dir, env_dataset, image_col, extension='mp4')

        import wandb
        videos[f'Eval videos/{video_prefix}/{env_name}'] = wandb.Video(video_path, format='mp4')

    return videos


def _log_to_wandb(args, config, dataset, info, evaluator):
    import wandb

    run = wandb.init(
        name=args.wandb_run_name or f'{config.app} eval',
        group=args.wandb_group or config.app,
        job_type=args.wandb_job_type,
        config=config.to_dict(),
        reinit=True,
    )

    log_data = _add_key_prefix(info, args.wandb_prefix)
    if args.wandb_video:
        video_prefix = args.wandb_video_prefix or args.wandb_prefix or config.app
        log_data |= _get_wandb_videos(evaluator, dataset, video_prefix)

    wandb.log(log_data)
    run.finish()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained policy from a config file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to a config file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model.safetensors')
    parser.add_argument('--wandb', action='store_true', help='log evaluation metrics to Weights & Biases')
    parser.add_argument('--wandb_prefix', type=str, default='Eval', help='metric prefix for wandb logs')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandb group name')
    parser.add_argument('--wandb_job_type', type=str, default='eval', help='wandb job type')
    parser.add_argument('--wandb_video', action=argparse.BooleanOptionalAction, default=True,
                        help='log one rollout video per evaluated environment to wandb')
    parser.add_argument('--wandb_video_prefix', type=str, default=None, help='video key prefix for wandb logs')
    args = parser.parse_args()

    config = Config.load(args.config_path)
    metadata = Metadata.from_hf(config.dataset.repo_ids[0], config.dataset.get('revision', 'main'))
    processor = DefaultProcessor(config, metadata)

    policy = get_policy(config, metadata, processor)
    checkpoint_path = args.checkpoint_path or get_best_model_checkpoint(config)
    policy.load_state_dict(load_file(checkpoint_path, device='cpu'))
    policy.to('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval()

    evaluator = AgentEvaluatorMultiEnv(policy, config)
    dataset, info = evaluator()
    pprint(info, sort_dicts=False)

    if args.wandb:
        _log_to_wandb(args, config, dataset, info, evaluator)


if __name__ == '__main__':
    main()
