import datasets
from pathlib import Path
import torch
from functools import cached_property
import pandas as pd
import numpy as np
from robo_utils.common import disable_hf_progress_bars, filter_dataset
from .utils import FeatureConfig, download_data, get_missing_values

class DataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        episode_ids: list[int],
        feature_configs: list[FeatureConfig] = None,
    ):
        '''
        Loads a dataset from a local path.
        Supports feature extraction at multiple steps (with `FeatureConfig`).
        Handles padding across episode boundaries (adds boolean padding masks for each feature).
        '''
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.episode_ids = episode_ids
        self._load_hf_dataset()
        self._set_feature_configs(feature_configs)

    @disable_hf_progress_bars()
    def _load_hf_dataset(self):
        '''Loads HF dataset and filters it with requested episodes only.'''
        # Load data dataset
        self.hf_dataset = datasets.load_dataset(
            'parquet', 
            data_dir=self.dataset_path, 
            split='train'
        )
        
        # Check that all requested episodes are present in the dataset
        missing = get_missing_values(self.hf_dataset.unique('episode_index'), self.episode_ids)
        if missing: raise FileNotFoundError(f'missing episodes: {missing}')
        
        # Filter dataset by requested episodes
        self.hf_dataset = filter_dataset(self.hf_dataset, self.episode_ids, 'episode_index')

        # Keep the native format and tensorize numeric columns manually in
        # __getitem__. datasets==4.1.1's torch formatter imports
        # torchvision.io.VideoReader, which is not available in newer
        # torchvision builds.

    def _set_feature_configs(self, feature_configs):
        # handle feature configs
        if feature_configs is not None:
            self.feature_configs = feature_configs
        else:
            features = list(self.hf_dataset.features)
            self.feature_configs = [FeatureConfig(name, [0]) for name in features]


    def __getitem__(self, index: int):
        '''
        Returns a dictionary of features for the given index, possibly including
        multiple steps (horizons). Handles non-video features,
        and adds padding masks if horizons cross episode boundaries or go out-of-bounds.

        This code benefits from Lazy column introduced in datasets==4.0.0.
        '''
        result = {}
        for feature in self.feature_configs:
            ids, pad_mask = self._get_horizon_info(index, feature)

            result[feature.name] = torch.as_tensor(self.hf_dataset[feature.name][ids])

            # Add padding mask if applicable
            if len(feature.horizon) > 1:
                result[f'{feature.name}_is_pad'] = pad_mask

            # Remove T dim if single step horizon was requested
            if len(feature.horizon) == 1:
                result[feature.name].squeeze_(0)

        return result

    def _get_horizon_info(self, index: int, feature: FeatureConfig):
        current_episode_id = self.hf_dataset['episode_index'][index]
        horizon_indices = index + feature.horizon

        # Clip to valid dataset bounds (avoid IndexError)
        min_idx, max_idx = 0, len(self.hf_dataset) - 1
        safe_indices = np.clip(horizon_indices, min_idx, max_idx).astype(int)
        
        # True where the original indices are out-of-bounds
        oob_mask = ( horizon_indices < min_idx) | (horizon_indices > max_idx)

        # True where the episode index does not match the current index.
        episode_ids = np.asarray(self.hf_dataset['episode_index'][safe_indices.tolist()])
        episode_mask = current_episode_id != episode_ids

        # Create padding mask - True where data is padded.
        pad_mask = torch.as_tensor(episode_mask | oob_mask)

        return safe_indices.tolist(), pad_mask

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        episode_ids: list[int],
        feature_configs: list[FeatureConfig] = None,
        revision = 'main',
    ):
        '''
        Factory: create dataset from a HuggingFace repo.
        First, tries to load local data; if missing, this will trigger downloading data.
        This avoids the need for slow HTTP check requests to see if the data is up to date
        (https://github.com/huggingface/datasets/issues/5499)).

        Examples:
        ---
        ```
        dataset = DataDataset.from_hf('username/repo_name', episode_ids=[0, 1])
        ```
        '''
        from .episode_dataset import DATASETS_HOME
        dataset_path = DATASETS_HOME / repo_id / 'data'

        try:
            ds = cls(dataset_path, episode_ids, feature_configs)
        except (FileNotFoundError, NotADirectoryError):
            download_data(repo_id, revision, 'data/', dataset_path.parent)
            ds = cls(dataset_path, episode_ids, feature_configs)

        ds.repo_id = repo_id
        ds.revision =revision
        return ds

    def __len__(self):
        return len(self.hf_dataset)
    
    @property
    def num_frames(self):
        return len(self.hf_dataset)

    @property
    def num_episodes(self):
        return len(self.hf_dataset.unique('episode_index'))


    @cached_property
    def episode_ranges(self):
        df = pd.DataFrame({'episode_index': self.hf_dataset['episode_index'][:]})
        df = df.sort_values(['episode_index']).reset_index(drop=True)
        return df.groupby('episode_index').agg(
            start_index=('episode_index', lambda x: x.index.min()),
            end_index=('episode_index', lambda x: x.index.max() + 1)
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'num_episodes={len(self.episode_ids)},\n'
            f'repo_name={self.dataset_path.parent.stem},\n'
            f'features={[feature.name for feature in self.feature_configs]})'
        )
