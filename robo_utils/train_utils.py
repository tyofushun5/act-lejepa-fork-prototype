from configs.training import Config
from robo_utils.dataset import EpisodeConcatDataset, Metadata
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn
from einops import rearrange
from models.policy_wrapper import PolicyWrapper, ModelWrapper
from transformers import TrainingArguments, set_seed, Trainer
from pathlib import Path

def _get_datasets_repo_ids(config: Config):
    repo_ids = config.dataset.get('repo_ids')

    # check for collection
    if repo_ids is None:
        from huggingface_hub import HfApi
        api = HfApi()
        collection_slug = config.datasets.collection_slug
        collection = api.get_collection(collection_slug)
        repo_ids = [item.item_id for item in collection.items if item.item_type == 'dataset']

    return repo_ids

def _make_feature_configs(config):
    '''Helper to create feature and video configs from config.feature_config.'''
    from robo_utils.dataset.utils import FeatureConfig
    feature_configs, video_configs = [], []
    for name, horizon in config.feature_config.items():
        feature = FeatureConfig(name, range(horizon))
        if 'image' in name:
            video_configs.append(feature)
        else:
            feature_configs.append(feature)
    return feature_configs, video_configs

def get_datasets(config: Config):
    '''Create and return train set and test sets.'''
    feature_configs, video_configs = _make_feature_configs(config)
    repo_ids = _get_datasets_repo_ids(config)
        
    train_set = EpisodeConcatDataset.from_hf(
        repo_ids=repo_ids,
        episode_ids=np.arange(*config.dataset.train_episodes_range),
        feature_configs=feature_configs,
        video_configs=video_configs,
        use_videos=config.dataset.use_videos,
        revision=config.dataset.revision,
    )

    test_set = EpisodeConcatDataset.from_hf(
        repo_ids=repo_ids,
        revision=config.dataset.revision,
        episode_ids=np.arange(*config.dataset.test_episodes_range),
        video_configs=video_configs,
        feature_configs=feature_configs,
        use_videos=config.dataset.use_videos,
    )

    return train_set, test_set

def get_dataloaders(config: Config):
    '''Creates and returns dataloader for train and test sets.'''
    train_set, test_set = get_datasets(config)

    conf = config.training_arguments
    train_loader = DataLoader(
        train_set, 
        batch_size = conf.per_device_train_batch_size,
        shuffle=True,
        num_workers=conf.dataloader_num_workers,
        drop_last=conf.dataloader_drop_last,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size = conf.per_device_eval_batch_size,
        shuffle=False,
        num_workers = conf.dataloader_num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader

class ImageProcessor:
    def __init__(self, img_size: int):
        from torchvision.transforms import v2
        self.resize = v2.Resize(img_size)
    
    def __call__(self, x: torch.Tensor):
        x = self.scale_image(x)
        x = self.to_channel_first(x)
        x = self.resize(x)
        return x
    
    def scale_image(self, x: torch.Tensor):
        is_float_input = 'float' in str(x.dtype)
        x = x.to(dtype=torch.float32)
        x  = x if is_float_input else x.div_(255.)
        return x

    @classmethod
    def to_channel_first(cls, x: torch.Tensor):
        if x.shape[-1] == 3:
            x = rearrange(x, '... h w c -> ... c h w').contiguous()
        return x
    
    @classmethod
    def to_channel_last(cls, x: torch.Tensor):
        if x.shape[-1] != 3:
            x = rearrange(x, '... c h w -> ... h w c').contiguous()
        return x

class DefaultProcessor:
    '''Processes dataset features - process images and normalize features.'''
    def __init__(self, config: Config, metadata: Metadata):
        self.image_transform = ImageProcessor(config.env.env_kwargs.img_size)

        self.normalize_keys = config.get('normalize_keys', [])
        if self.normalize_keys:
            from robo_utils.normalize import StatsNormalize
            self.normalize_transform = StatsNormalize(metadata.stats, config.normalize_keys)
        else: 
            self.normalize_transform = lambda x: x # identity fun if nothing to normalize
        
    def denormalize(self, inputs: dict):
        if self.normalize_keys:
            return self.normalize_transform.unapply(inputs)
        else:
            return inputs

    def __call__(self, inputs: dict):
        inputs = dict(inputs) # shallow copy

        # Process images
        for key in inputs.keys():
            if 'image' in key and 'is_pad' not in key:
                inputs[key] = self.image_transform(inputs[key])

        # Normalize actions, states, etc.
        inputs = self.normalize_transform(inputs)

        return inputs

class DefaultProcessorWithLabels(DefaultProcessor):
    def __init__(self, config, metadata, labels_name=None):
        '''
        Apply transformations (e.g., image scaling).
        Add `labels` key (provided during training, not rollout).
        Return a dict of transformed inputs (and possibly labels).

        Important: both labels and inputs are processed the same
        way to calculate loss. E.g., normalized actions or images to float.
        '''
        super().__init__(config, metadata)
        self.labels_name = labels_name
    
    def __call__(self, inputs: dict):
        inputs = super().__call__(inputs)
        inputs['labels'] = inputs.get(self.labels_name)
        return inputs

class ProcessorWrapper(ModelWrapper):
    '''
    Apply processing before forward pass. This way
    transformations are efficient and can be applied on GPUs.
    '''
    def __init__(self, model, processor):
        super().__init__(model)
        self.processor = processor

    def forward(self, *args, **kwargs):
        kwargs = self.processor(kwargs)
        return self.model.forward(*args, **kwargs)

def get_best_model_checkpoint(config: Config):
    '''Return the path to the best model checkpoint from trainer state.'''
    trainer_state_conf = Config.from_json_file(f'logs/training/{config.app}/trainer_state.json')
    best_model_checkpoint = trainer_state_conf.best_model_checkpoint
    best_model_checkpoint = Path(best_model_checkpoint) / 'model.safetensors'
    return best_model_checkpoint

def get_model_class(class_name: str) -> nn.Module:
    '''Dynamically load a model or a policy class based on config and return it.'''
    try:
        import models
        model_class = getattr(models, class_name)
    except AttributeError:
        raise ValueError(f'Class {class_name} not found.')
    return model_class

def get_policy(config: Config, metadata: Metadata, processor=None) -> PolicyWrapper:
    model_class = get_model_class(config.model.class_name)
    model_config = model_class.config_class(
        action_dim = metadata.info.features['action'].shape[0],
        state_dim = metadata.info.features['observation.state'].shape[0],
        **config.model
    )
    model= model_class(model_config)

    model = ProcessorWrapper(model, processor)

    policy_class = get_model_class(config.model.policy_name)
    policy = policy_class(model)
    
    return policy

def default_train_loop(config: Config, callbacks: list):
    # set seed
    set_seed(config.training_arguments.seed)

    train_set, test_set = get_datasets(config)
    processor = DefaultProcessorWithLabels(config, train_set.metadata, 'action')

    # create a policy model
    policy = get_policy(config, train_set.metadata, processor)

    # create a trainer
    args = TrainingArguments(
        **config.training_arguments,
        remove_unused_columns=False,
        output_dir=f'logs/training/{config.app}',
        run_name=config.app,
    )
    trainer = Trainer(
        policy, args, 
        train_dataset=train_set, eval_dataset=test_set
    )
    make_deterministic()
    for cb in callbacks: trainer.add_callback(cb(trainer, config))
    
    # train
    trainer.train()
    trainer.save_state()
    return trainer

def make_deterministic():
    '''Handle Conv layer funkiness.'''
    import os, torch, platform
    if platform.system() == "Linux":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        torch.use_deterministic_algorithms(True)

