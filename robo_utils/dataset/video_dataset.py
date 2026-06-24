import torch
from pathlib import Path
import datasets
from torchcodec.decoders import VideoDecoder
from functools import cached_property
from tqdm import tqdm
import numpy as np
from robo_utils.common import disable_hf_progress_bars, filter_dataset
from .utils import FeatureConfig, download_data, get_missing_values

class VideoDataset:

    def __init__(
        self,
        dataset_path: str | Path,
        episode_ids: list[int],
        feature_configs: list[FeatureConfig] = None,
    ):
        '''
        Loads a video dataset from a local path.
        '''
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.episode_ids = episode_ids

        # Check if we have zip files and unzip them
        self._unzip_videos_if_needed(dataset_path)
        
        self._load_hf_dataset()
        self._set_feature_configs(feature_configs)

    @disable_hf_progress_bars()
    def _load_hf_dataset(self):
        '''Loads HF dataset.'''
        # Load dataset
        self.hf_dataset = datasets.load_dataset(
            'videofolder', 
            data_dir=self.dataset_path, 
            split='train', 
            drop_labels=False
        )
        # Add episode index
        self.hf_dataset = self.hf_dataset.map(self._add_episode_index_col)

        # Add camera keys/names
        self.hf_dataset = self.hf_dataset.map(self._add_camera_key)

        # Check that all requested episodes are present in the dataset
        missing = get_missing_values(self.hf_dataset.unique('episode_index'), self.episode_ids)
        if missing: raise FileNotFoundError(f'missing episodes: {missing}')

        # Filter dataset by requested episodes
        self.hf_dataset = filter_dataset(self.hf_dataset, self.episode_ids, 'episode_index')

    def _set_feature_configs(self, feature_configs):
        # handle feature configs
        if feature_configs is not None:
            self.feature_configs = feature_configs
        else:
            self.feature_configs = [FeatureConfig(name, [0]) for name in self.camera_keys]

    def _add_episode_index_col(self, row: dict):
        '''Extract episode index from a video filename and add as a new column.'''
        import re
        filename = Path(row['video'].metadata.path).stem
        match = re.search(r'(\d+)', filename)
        episode_idx = int(match.group(1))
        row['episode_index'] = episode_idx
        return row

    def _add_camera_key(self, row: dict):
        row['camera_key'] = self.hf_dataset.features['label'].int2str(row['label'])
        return row

    def get_frames_at(self, episode_index: int, start_frame_index: int):
        result = {}
        for feature in self.feature_configs:
            video = self.get_video(episode_index, feature.name)
            safe_ids, pad_mask = self._get_horizon_info(start_frame_index, feature, video)
            frames = video.get_frames_at(safe_ids)
            result[feature.name] = frames.data

            # Add padding mask if applicable
            if len(feature.horizon) > 1:
                result[f'{feature.name}_is_pad'] = pad_mask

            # Remove T dim if single step horizon was requested
            if len(feature.horizon) == 1 and isinstance(result[feature.name], torch.Tensor):
                result[feature.name].squeeze_(0)
                
        return result

    def _get_horizon_info(self, start_frame_index: int, feature: FeatureConfig, video: VideoDecoder):
        horizon_indices = start_frame_index + feature.horizon
        min_idx, max_idx = 0, len(video) - 1
        safe_indices = np.clip(horizon_indices, min_idx, max_idx)
        
        # True where the original indices are out-of-bounds
        oob_mask = (horizon_indices < min_idx) | (horizon_indices > max_idx)
        oob_mask = torch.asarray(oob_mask)
        return safe_indices, oob_mask

    def get_video(self, episode_index: int, camera_key: str) -> VideoDecoder:
        '''Return video for episode and camera.'''
        episode_index = int(episode_index)
        hf_row = int(self.lookup.loc[(episode_index, camera_key)]['hf_row'])
        return self.hf_dataset[hf_row]['video']

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
        dataset = EpisodeVideoDataset.from_hf('username/repo_name', episode_ids=[0, 1])
        ```
        '''
        from .episode_dataset import DATASETS_HOME
        dataset_path = DATASETS_HOME / repo_id / 'videos'
        
        try:
            ds = cls(dataset_path, episode_ids, feature_configs)
        except (FileNotFoundError, NotADirectoryError):
            allow_patterns = [f'videos/**/episode_{i:06d}.mp4' for i in episode_ids]
            allow_patterns += [f'videos/**/*.zip'] # allow zipped files
            download_data(repo_id, revision, allow_patterns, dataset_path.parent)
            ds = cls(dataset_path, episode_ids, feature_configs)
        
        ds.repo_id = repo_id
        ds.revision = revision
        return ds

    def __len__(self):
        return self.num_episodes
    
    @cached_property
    def lookup(self):
        '''DataFrame index for fast (episode, camera) -> row lookup.'''
        df = self.hf_dataset.to_pandas()
        df['hf_row'] = df.index
        df.set_index(['episode_index', 'camera_key'], inplace=True)
        return df

    @cached_property
    def num_frames(self):
        return sum([v.metadata.num_frames for v in self.hf_dataset['video']])

    @property
    def num_episodes(self):
        return len(self.hf_dataset.unique('episode_index'))
    
    @property
    def camera_keys(self):
        return self.hf_dataset.features['label'].names
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'num_episodes={len(self.episode_ids)},\n'
            f'camera_keys={self.camera_keys},\n'
        )

    @staticmethod
    def _unzip_videos_if_needed(dataset_path: Path):
        '''Unzip video files if they exist, then remove zip files.'''
        import zipfile
        
        videos_dir = dataset_path
        if not videos_dir.exists(): return
        
        zip_files = list(videos_dir.rglob('*.zip'))
        if not zip_files: return  # No zip files found
        
        for zip_path in tqdm(zip_files, desc='Unzipping videos'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to the same directory as the zip file
                extract_dir = zip_path.parent
                zip_ref.extractall(extract_dir)
            
            # Remove the zip file after successful extraction
            zip_path.unlink()
        
        print('Unzipping completed.')
