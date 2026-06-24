import torch
from pathlib import Path
from functools import cached_property
from tqdm.contrib.concurrent import thread_map

from huggingface_hub.constants import HF_HOME
DATASETS_HOME = Path(HF_HOME) / 'episode_dataset'

from .utils import FeatureConfig

from .data_dataset import DataDataset
from .video_dataset import VideoDataset
from .metadata import Metadata

class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dataset: DataDataset, 
        video_dataset: VideoDataset, 
        metadata: Metadata
    ):
        super().__init__()
        self.data_dataset = data_dataset
        self.video_dataset = video_dataset
        self.metadata = metadata

    def __len__(self):
        return len(self.data_dataset)
    
    @property
    def num_frames(self):
        return self.data_dataset.num_frames
    
    @property
    def num_episodes(self):
        return self.data_dataset.num_episodes
    
    def __getitem__(self, index: int):
        '''
        Returns a dictionary of features for the given index, possibly including
        multiple steps (horizons). Handles non-video features, video features,
        and adds padding masks if horizons cross episode boundaries or go out-of-bounds.
        '''
        # Non-video features
        result = self.data_dataset[index]

        if self.video_dataset is None: 
            return result

        # Video features
        episode_index = self.data_dataset.hf_dataset['episode_index'][index]
        frame_index = self.data_dataset.hf_dataset['frame_index'][index]
        result |= self.video_dataset.get_frames_at(episode_index, frame_index)
                
        return result

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        episode_ids: list[int],
        feature_configs: list[FeatureConfig] = None,
        video_configs: list[FeatureConfig] = None,
        use_videos = True,
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
        # Load data dataset (parquet)
        data_dataset = DataDataset.from_hf(repo_id, episode_ids, feature_configs, revision)

        # Load video dataset
        if use_videos:
            video_dataset = VideoDataset.from_hf(repo_id, episode_ids, video_configs, revision)
        else:
            video_dataset = None

        # Load metadata (optional)
        try:
            metadata = Metadata.from_hf(repo_id, revision)
        except: 
            metadata = None
        
        return cls(data_dataset, video_dataset, metadata)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'num_episodes={len(self.episode_ids)},\n'
            f'repo_name={self.dataset_path.stem},\n'
        )

    @cached_property
    def dataset_path(self):
        return DATASETS_HOME / self.data_dataset.dataset_path.parent

    @property
    def episode_ids(self):
        return self.data_dataset.episode_ids

    @cached_property
    def episode_ranges(self):
        return self.data_dataset.episode_ranges


class EpisodeConcatDataset(torch.utils.data.ConcatDataset):
    '''
    Concatenates multiple episode datasets into a single dataset.
    Useful for combining data from several HuggingFace repos or local sources.
    Provides unified access to metadata, frame/episode counts, and feature configs.
    
    For DataLoader, make sure that all datasets have the same feature config.
    '''
    def __init__(self, datasets: list[EpisodeDataset]):
        super().__init__(datasets)
        self.repo_names = [ds.dataset_path.stem for ds in datasets]

    @classmethod
    def from_hf(
        cls,
        repo_ids: list[str],
        episode_ids: list[int],
        feature_configs: list[FeatureConfig] = None,
        video_configs: list[FeatureConfig] = None,
        use_videos: bool = True,
        revision: str = 'main',
    ):
        def _load(repo_id: str):
            return EpisodeDataset.from_hf(repo_id, episode_ids, feature_configs, video_configs, use_videos, revision)
        desc = f'Loading {len(repo_ids)} datasets, {len(episode_ids)} episodes each'
        datasets_list = thread_map(_load, repo_ids, desc=desc, max_workers=8)
        return cls(datasets_list)
    
    @property
    def num_frames(self):
        return sum(d.num_frames for d in self.datasets)
    
    @property
    def num_episodes(self):
        return sum(d.num_episodes for d in self.datasets)
    
    @property
    def metadata(self):
        return self.datasets[0].metadata

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'  repo_names={self.repo_names},\n'
            f'  num_frames={self.num_frames},\n'
            f'  num_episodes={self.num_episodes},\n'
            f')'
        )
