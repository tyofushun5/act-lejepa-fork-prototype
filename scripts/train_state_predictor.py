# Experiment: Evaluating Ability to Reconstruct Observation Sequences

# Description:
# Test whether ACT-JEPA's learned representations can reconstruct 
# proprioceptive states (e.g., robot arm trajectories), extending beyond action 
# sequence prediction. The context encoder is frozen, and a new decoder head is 
# trained for this task.

# Evaluation Metrics:
# - Root Mean Squared Error (RMSE): Measures error magnitude.
# - Absolute Trajectory Error (ATE): Assesses trajectory alignment.

from init import init; init()
import argparse

from configs.training import Config
from pathlib import Path
from robo_utils.train_utils import (
    DefaultProcessorWithLabels,
    Metadata,
    ProcessorWrapper,
    Trainer,
    TrainingArguments,
    get_best_model_checkpoint,
    get_datasets,
    get_policy,
    make_deterministic,
    set_seed,
)

def main():
    parser = argparse.ArgumentParser(description='Train model with config file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to a config file')
    parser.add_argument('--base_config_path', type=str, required=True, help='Path to a base config file')
    args = parser.parse_args()
    config = Config.load(args.config_path)
    base_config = Config.load(args.base_config_path)
    callback_names = getattr(config, 'callbacks')
    from robo_utils import callbacks as callbacks
    callback_list = [getattr(callbacks, name) for name in callback_names]
    train_loop(config, base_config, callback_list)

    import wandb
    wandb.finish()


def train_loop(config: Config, base_config: Config, callbacks: list, encoder=None):
    # get the encoder
    if encoder is None:
        from safetensors.torch import load_file
        checkpoint_path = Path(get_best_model_checkpoint(base_config))
        state_dict = load_file(checkpoint_path, device='cpu')
        
        metadata = Metadata.from_hf(base_config.dataset.repo_ids[0])
        policy = get_policy(base_config, metadata, processor=None)
        policy.load_state_dict(state_dict)
        encoder = policy.model.encoder

    # set seed
    set_seed(config.training_arguments.seed)

    train_set, test_set = get_datasets(config)
    # get processor where labels (what we predict) are states, not actions
    processor = DefaultProcessorWithLabels(base_config, train_set.metadata, 'observation.state')

    # create a state predictor
    from models import state_predictor
    model_config = state_predictor.StatePredictorConfig(
        state_dim = train_set.metadata.info.features['observation.state'].shape[0],
        **config.model
    )
    model = state_predictor.StatePredictorModel(model_config, encoder)
    model = ProcessorWrapper(model, processor)

    # create a trainer
    args = TrainingArguments(
        **config.training_arguments,
        remove_unused_columns=False,
        output_dir=f'logs/training/{config.app}',
        run_name=config.app,
    )
    trainer = Trainer(
        model, args, 
        train_dataset=train_set, eval_dataset=test_set, 
    )
    make_deterministic()
    for cb in callbacks: trainer.add_callback(cb(trainer, config))
    
    # train
    trainer.train()
    trainer.save_state()
    return trainer

if __name__ == '__main__':
    # python -m scripts.train_state_predictor --config_path configs/state_predictor.yaml --base_config_path configs/act-jepa.yaml
    # python -m scripts.train_state_predictor --config_path configs/pusht/state_predictor.yaml --base_config_path "configs/pusht/act-jepa causal.yaml"
    main()
