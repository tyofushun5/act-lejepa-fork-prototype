from transformers import TrainerCallback, PretrainedConfig, Trainer
import wandb

from robo_utils.common import HFDataset, Path, datasets, save_episode_video
from time import time
import pandas as pd
from copy import deepcopy

class AgentEvaluatorCallback(TrainerCallback):
    '''Callback for evaluating an agent using MultiAgentEvaluator.'''
    def __init__(self, trainer: Trainer, config: PretrainedConfig):
        self.rollout_steps = config.env.get('rollout_steps', 1000000)
        self.rollout_delay = config.env.get('rollout_delay', 0)
        self.metric_key = 'Rollout/solved %'
        self.info = {} # will be populated

        self.trainer = trainer
        policy = trainer.model
        from robo_utils.agent_evaluator_multi_env import AgentEvaluatorMultiEnv
        self.evaluator = AgentEvaluatorMultiEnv(policy, config)

        import custom_envs # trigger import environments

    def on_step_end(self, args, state, control, **kwargs):
        '''
        Performs agent evaluation and logging at rollout intervals.
        Although this method is named on_step_end due to TrainerCallback
        requirements, it essentially implements rollout evaluation logic.
        '''
        if not state.is_world_process_zero:
            return
        if not self.should_rollout():
            return

        start_time = time()
        dataset, info = self.evaluator()
        info['Rollout/sum time (s)'] = time() - start_time

        # Check if a new best metric is found
        self.should_save(metric_value=info[self.metric_key])

        # Log to wandb        
        if 'wandb' in args.report_to:
            wandb_video = {'video': self.get_wandb_video(dataset)}
            wandb.log((info | wandb_video), step=state.global_step)
        
        # self.save_dataset_to_disk(dataset)
        self.info = info
        
    def save_dataset_to_disk(self, dataset: HFDataset):
        from .dataset.save_utils import save_dataset_pipeline
        dataset_path = f'logs/wandb/rollout_dataset_{self.trainer.state.global_step}'
        save_dataset_pipeline(dataset, dataset_path, fps=30)

    def get_wandb_video(self, dataset):
        '''Create and return a wandb video for logging.'''
        img_col_name = next((n for n, t in dataset.features.items()
                            if isinstance(t, datasets.features.Image)), None)
        
        if img_col_name:
            extension = 'mp4'
            video_path = save_episode_video('logs/wandb', dataset, img_col_name, extension=extension)
            ep_video = wandb.Video(video_path, format=extension)
            return ep_video
    
    def should_rollout(self):
        '''Check if it's time to rollout and evaluate the agent.'''
        step = self.trainer.state.global_step
        if step <= self.rollout_delay:
            return False
        if step == 1:
            return True
        if step % self.rollout_steps == 0:
            return True
        return False
    
    def should_save(self, metric_value):
        '''
        Check if the current metric value is the best so far and update Trainer state.
        If the metric is 'solved %' and a new best is achieved, update best_metric and best_global_step.
        If using SaveStrategy.BEST, signal the Trainer to save the checkpoint.
        Args:
            metric_value (float): The current value of the metric being tracked (e.g., 'solved %').
        '''
        # Initialize best_metric if it hasn't been set yet
        if self.trainer.state.best_metric is None:
            self.trainer.state.best_metric = float('-inf')

        # If this evaluation produced a new best metric, update state and trigger save if needed
        if metric_value > self.trainer.state.best_metric:
            self.trainer.state.best_metric = metric_value

            # Save the best model, no matter the strategy
            self.trainer.state.best_global_step = self.trainer.state.global_step
            self.trainer.control.should_save = True

            print(f'New best {self.metric_key}: {metric_value} at step {self.trainer.state.global_step}')


class SaveBestCheckpointCallback(TrainerCallback):
    '''
    Loads and saves the best model checkpoint at training end.

    Note:
    Unlike HF's `load_best_model_at_end`, this works with custom
    evaluation (e.g., agent evaluators) not supported by standard
    evaluation logic.
    '''
    def __init__(self, trainer: Trainer, config=None):
        self.trainer = trainer

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if state.best_model_checkpoint is None:
            raise RuntimeError('No best_model_checkpoint found.')

        print(f'Loading best model from: {state.best_model_checkpoint}')
        self.trainer._load_best_model()

        save_path = Path(args.output_dir) / 'best-checkpoint'
        print(f'Saving best model to: {save_path}')
        self.trainer.save_model(save_path)


class EmaUpdateCallback(TrainerCallback):
    '''
    Uses EMA to update weights of the target encoder.
    Note that if using grad accum, updating target encoder should happen 
    once the gradients are synced and the model's weights are updated.
    In HF trainer, this happens on the on_step_end event.
    '''
    def __init__(self, trainer: Trainer, config=None):
        super().__init__()
        self.trainer = trainer
        # Get the total number of optimizer updates. Note:
        # max_steps is the number of optimizer steps and already accounts 
        # for gradient_accumulation_steps. No need to divide by gradient_accumulation_steps.
        total_num_updates = trainer.args.max_steps 
        self.ema_momentum = config.ema_start
        self.ema_step = (config.ema_end - config.ema_start) / total_num_updates

    def on_optimizer_step(self, args, state, control, **kwargs):
        # Apply EMA - update the target encoder weights
        self.trainer.model.model.update_target_encoder(m=self.ema_momentum)

        # Update EMA momentum (and clip so we don't exceed valid range)
        self.ema_momentum += self.ema_step 
        self.ema_momentum = min(self.ema_momentum, 1)    

    def on_step_end(self, args, state, control, **kwargs):
        # Log EMA to wandb
        if not state.is_world_process_zero or 'wandb' not in args.report_to or not control.should_log:
            return
        wandb.log({'train/ema': self.ema_momentum}, state.global_step)


class HardTargetUpdateCallback(TrainerCallback):
    '''
    Copies the context encoder weights into the target encoder after each
    optimizer step. This disables EMA target updates while keeping the two
    encoders synchronized throughout training.
    '''
    def __init__(self, trainer: Trainer, config=None):
        super().__init__()
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer.model.model.copy_context_to_target_encoder()

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or 'wandb' not in args.report_to or not control.should_log:
            return
        wandb.log({'train/target_encoder_hard_sync': 1}, state.global_step)


class CollectLossCallback(TrainerCallback):

    def __init__(self, trainer: Trainer, config=None):
        self.trainer = trainer
        self.auxiliary_losses = []
        self.eval_losses = []
    
    def _get_auxiliary_loss_values(self):
        # NOTE: this works only on a single process, not on distributed.
        # the model might be wrapped
        obj = self.trainer.model
        while hasattr(obj, 'model'):
            obj = obj.model
        loss_values = {k: getattr(obj, k) for k in dir(obj) if k.endswith('_loss')}
        return loss_values

    def on_step_end(self, args, state, control, **kwargs):
        loss_values = self._get_auxiliary_loss_values()
        self.auxiliary_losses.append(loss_values)
    
    def on_evaluate(self, args, state, control, logs = None, **kwargs):
        if logs and 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])


class LogAuxiliaryLossCallback(CollectLossCallback):

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or 'wandb' not in args.report_to or not control.should_log:
            return
        loss_values = self._get_auxiliary_loss_values()
        add_key_prefix = lambda d, prefix: {f'{prefix}/{k}': v for k, v in d.items()}
        loss_values = add_key_prefix(loss_values, 'train')
        wandb.log(loss_values, step=state.global_step)


import torch
import gc
 
class _PreserveTorchRNG:
    '''Preserve and restore Torch RNG state (CPU and all CUDA devices).'''
    def __enter__(self):
        self.cpu_state = torch.get_rng_state()
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.cuda_states = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.set_rng_state(self.cpu_state)
        if self.has_cuda:
            torch.cuda.set_rng_state_all(self.cuda_states)
        return False

class StatePredictorCallback(TrainerCallback):
    '''Periodically clones the current model and runs a short fine-tuning (probe) loop to track fine-tuning losses during pre-training.'''
    def __init__(self, trainer: Trainer, config=None):
        super().__init__()
        self.trainer: Trainer = trainer
        self.config = config
        assert self.config.get('state_predictor') is not None
        # Make sure that wandb doesn't try to create new instance in the callback.
        self.config.state_predictor.training_arguments.report_to = [] 
    # No need to keep historical probe callbacks/results; we'll log and discard.

    def should_run(self) -> bool:
        every_n_steps = self.config.state_predictor_every_n_steps
        step = self.trainer.state.global_step
        if step == 1:
            return True
        if step % every_n_steps == 0:
            return True
        return False

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.should_run():
            return
        # Start probe training and collect losses
        from scripts.train_state_predictor import train_loop
        encoder = deepcopy(self.trainer.model.model.encoder)
        
        print('State Predictor Training Starting')
        # Preserve Torch RNG state so probe training doesn't perturb the main loop
        with _PreserveTorchRNG():
            trainer = train_loop(self.config.state_predictor, self.config, [CollectLossCallback], encoder)

        # Extract results from the probe callback, then break references to free GPU memory
        cb = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, CollectLossCallback))
        results = {
            'auxiliary_losses': list(cb.auxiliary_losses),
            'eval_losses': list(cb.eval_losses),
        }
        
        # check if the original (not callback) trainer is reporting to wandb
        if 'wandb' in self.trainer.args.report_to:
            self._log_to_wandb(results)

        # cleanup
        del cb
        del trainer
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_to_wandb(self, results: dict):
        loss_data = pd.DataFrame(results['auxiliary_losses'])
        if not loss_data.empty:
            # Log min (best) loss per column
            for loss_name in loss_data.columns:
                min_value = loss_data[loss_name].min()
                wandb.log({f'state_predictor/{loss_name}': min_value}, self.trainer.state.global_step)

        # Log min eval loss if available
        if results.get('eval_losses'):
            min_eval_loss = min(results['eval_losses'])
            wandb.log({'state_predictor/eval_loss': min_eval_loss}, self.trainer.state.global_step)


class ActionPredictorCallback(TrainerCallback):
    '''Periodically clone the current model and run a short fine-tuning (or probe) loop to track losses'''
    def __init__(self, trainer: Trainer, config=None):
        super().__init__()
        self.trainer: Trainer = trainer
        self.config = config
        assert self.config.get('action_predictor') is not None
        # Make sure that wandb doesn't try to create new instance in the callback.
        self.config.action_predictor.training_arguments.report_to = [] 
    
    def should_run(self) -> bool:
        every_n_steps = self.config.action_predictor_every_n_steps
        step = self.trainer.state.global_step
        if step == 1:
            return True
        if step % every_n_steps == 0:
            return True
        return False
    
    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.should_run():
            return
        # Start probe training and collect losses
        from scripts.train_action_predictor import train_loop
        encoder = deepcopy(self.trainer.model.model.context_encoder)

        print('Action Predictor Training Starting')
        # Preserve Torch RNG state so probe training doesn't perturb the main loop
        with _PreserveTorchRNG():
            trainer = train_loop(self.config.action_predictor, [CollectLossCallback, AgentEvaluatorCallback], encoder)

        # Extract loss
        results = {}
        final_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log][-1]
        results['action_predictor/reconstruction_loss'] = final_loss
        print(final_loss)

        # Extract rollout results
        cb = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, AgentEvaluatorCallback))
        if cb:
            results['rollout_results'] = cb.info

        # check if the original (not callback) trainer is reporting to wandb
        if 'wandb' in self.trainer.args.report_to:
            self._log_to_wandb(results)
            

    def _log_to_wandb(self, results: dict):
        
        key = 'action_predictor/reconstruction_loss'
        wandb.log({key: results[key]}, self.trainer.state.global_step)

        # Log best rollout
        rollout_results = results.get('rollout_results', None)
        if rollout_results:
            wandb.log((rollout_results), step=self.trainer.state.global_step)
        
