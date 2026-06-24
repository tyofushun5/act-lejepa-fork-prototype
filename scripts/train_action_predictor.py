# Experiment: Investigating JEPA's Generalization to Action Sequences

# Description:
# This script explores whether JEPA contributes to a policy's performance 
# by improving generalization to action sequences. The experiment consists 
# of two stages:
# 1. Pretraining the model to predict an abstract observation sequence.
# 2. Fine-tuning the model to predict an action sequence.

# Objective:
# Evaluate if pretraining on abstract observation prediction enables 
# JEPA to learn transferable representations that enhance 
# action sequence prediction.

# Does predicting abstract observation sequences generalize to predicting action sequences?

from init import init; init()
import argparse

from configs.training import Config
from robo_utils.train_utils import (
    DefaultProcessorWithLabels,
    ProcessorWrapper,
    Trainer,
    TrainingArguments,
    default_train_loop,
    get_datasets,
    make_deterministic,
    set_seed,
)

def main():
    parser = argparse.ArgumentParser(description='Train model with config file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to a config file')
    args = parser.parse_args()
    
    from robo_utils import callbacks as callbacks
    
    config = Config.load(args.config_path)
    callback_names = getattr(config, 'callbacks')
    callback_list = [getattr(callbacks, name) for name in callback_names]
    default_train_loop(config, callback_list)

    import wandb
    wandb.finish()

def train_loop(config: Config, callbacks: list, encoder=None):
    # set seed
    set_seed(config.training_arguments.seed)

    train_set, test_set = get_datasets(config)
    # get processor where labels (what we predict) are actions
    processor = DefaultProcessorWithLabels(config, train_set.metadata, 'action')

    from models import action_predictor    
    model_config = action_predictor.ActConfig(
        action_dim = train_set.metadata.info.features['action'].shape[0],
        state_dim = train_set.metadata.info.features['observation.state'].shape[0],
        **config.model
    )
    model = action_predictor.ActionPredictorModel(model_config, encoder)
    model = ProcessorWrapper(model, processor)
    model = action_predictor.ActionChunkingPolicy(model)

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
    main()
