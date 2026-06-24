
import json
from pathlib import Path
import torch
import numpy as np
import datasets
from datasets import Dataset as HFDataset
import zipfile
from robo_utils.common import save_episode_videos
from .episode_dataset import DATASETS_HOME

def save_dataset_pipeline(dataset: HFDataset, repo_id: str, fps: int):
    '''Save dataset, video dataset, and metadata for a given dataset.'''

    dataset_path = DATASETS_HOME / repo_id
    video_path = dataset_path / 'videos'
    meta_path = dataset_path / 'meta'
    data_path = dataset_path / 'data'

    # Save episode videos
    video_cols = [name for name, type in dataset.features.items() if isinstance(type, datasets.features.Image)]
    save_episode_videos(video_path, dataset, video_cols, fps=fps)
    
    # Save info & Build info dict before removing video column, as want auto build features
    info_dict = build_info_json(dataset, repo_id, fps)
    save_info_json(meta_path, info_dict)

    # Remove video columns from dataset
    dataset = dataset.remove_columns(video_cols)

    # Save stats (without images)
    save_stats_json(meta_path, dataset)

    # Save data
    dataset.to_parquet(data_path / 'file-000.parquet') # save data

    # Zip videos: find all subdirs with mp4s and zip/shard them in place
    zip_mp4s_in_leaf_dirs(video_path)

    return dataset

def push_dataset_to_hub(repo_id: str, dataset_path: Path, branch_name: str = 'v3.5', private = True):
    '''Create repo if needed, upload to main, then create a new branch from main.'''
    from huggingface_hub import HfApi
    hub_api = HfApi()
    # Create repo if it doesn't exist
    hub_api.create_repo(repo_id=repo_id, repo_type='dataset', private=private, exist_ok=True)
    # Upload to main branch
    hub_api.upload_large_folder(
    repo_id=repo_id,
    folder_path=dataset_path,
    repo_type='dataset',
    revision='main'
    )
    # Create new branch from main
    hub_api.create_branch(repo_id=repo_id, repo_type='dataset', branch=branch_name)


########################## Save utils ##########################

def describe_feature(feature: np.ndarray | list):
    '''Return mean, std, min, max for each column of a feature.'''
    tensor = torch.as_tensor(feature, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    stats = {
        'mean': tensor.mean(0).tolist(),
        'std': tensor.std(0).tolist(),
        'min': tensor.min(0).values.tolist(),
        'max': tensor.max(0).values.tolist(),
    }
    return stats

def build_info_json(dataset: HFDataset, repo_id: str, fps: int, robot_type='Sawyer'):
    '''Build info.json structure for the dataset.'''
    # Auto build features
    dataset = dataset.with_format('np')
    row: dict = dataset[0]
    features = {}
    for name, value in row.items():
        features[name] = {'dtype': str(value.dtype), 'shape': value.shape, 'names': None}

    # Compute dynamic values
    total_episodes = len(dataset.unique('episode_index'))
    total_frames = len(dataset)
    total_tasks = 1
    # Count video files in the videos directory using pathlib
    dataset_path = DATASETS_HOME / repo_id
    videos_dir = dataset_path / 'videos'
    total_videos = sum(1 for _ in videos_dir.rglob('*.mp4'))
    splits = {'train': f'0:{total_episodes}'}

    info = {
        'codebase_version': 'v3.0',
        'robot_type': robot_type,
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'total_tasks': total_tasks,
        'total_videos': total_videos,
        'fps': fps,
        'splits': splits,
        'features': features
    }
    return info

def save_stats_json(meta_path: Path, dataset: HFDataset):
    '''Save statistics for all features to stats.json.'''
    meta_path.mkdir(parents=True, exist_ok=True)
    all_stats = {col_name: describe_feature(dataset[col_name]) for col_name in dataset.features}
    stats_path = meta_path / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)

def save_info_json(meta_path: Path, info):
    '''Save evaluation info to info.json.'''
    meta_path.mkdir(parents=True, exist_ok=True)
    info_path = meta_path / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)


def zip_dir_in_shards(folder_path: Path, max_shard_size: int, suffix='*'):  # formerly zip_videos
    '''Shard non-zip files under folder_path into <stem>_XYZ.zip archives (size-limited) and remove originals.'''
    # Collect all candidate files (exclude existing zips) in deterministic order
    files = sorted(f for f in folder_path.rglob(suffix) if f.is_file() and f.suffix != '.zip')
    if not files:
        return 0

    limit = max_shard_size * 1024 * 1024 # in bytes
    zf = None              # current open ZipFile
    zip_size = 0           # bytes accumulated in current shard
    zip_count = 0          # number of zip shards created

    for f in files:
        sz = f.stat().st_size
        # Open a new shard if none yet or adding file would exceed size limit
        if (zf is None) or (zip_size + sz > limit):
            if zf:
                zf.close()
            zf = zipfile.ZipFile(folder_path / f'{folder_path.name}_{zip_count:03d}.zip', 'w', zipfile.ZIP_DEFLATED)
            zip_size = 0
            zip_count += 1
        # Add file (relative path preserves folder structure) then delete original
        zf.write(f, f.relative_to(folder_path))
        zip_size += sz
        f.unlink()

    # Finalize last shard
    if zf:
        zf.close()
    print(f'Created {zip_count} zip files from {len(files)} files')
    return zip_count

def zip_mp4s_in_leaf_dirs(root: Path, max_shard_size: int = 500):
    '''Find all subdirs under root containing mp4s, zip their mp4s into shards in place, and remove originals.'''
    # Walk all subdirs, deepest first
    for subdir in sorted([p for p in root.rglob('') if p.is_dir()], key=lambda p: -len(p.parts)):
        if any(subdir.glob('*.mp4')):
            zip_dir_in_shards(subdir, max_shard_size, '*.mp4')