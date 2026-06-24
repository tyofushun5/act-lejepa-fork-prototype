from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import datasets
from datasets import Dataset as HFDataset
import rerun as rr
from pathlib import Path
from tqdm import tqdm
import uuid
from collections.abc import Iterable

def pprint_batch(batch: dict):
    from pprint import pprint
    res = {}
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            res[k] = f'{list(v.shape)} {v.dtype}'
        elif isinstance(v, list):
            res[k] = set(v)
        else:
            res[k] = v
    pprint(res, sort_dicts=False)


def pprint_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


class disable_hf_progress_bars:
    '''
    Context manager and decorator to temporarily disable Hugging Face Hub 
    and Hugging Face Datasets progress bars.
    '''
    def __enter__(self):
        from huggingface_hub.utils import disable_progress_bars, are_progress_bars_disabled
        from datasets.utils.logging import is_progress_bar_enabled, disable_progress_bar
        self._was_hub_disabled = are_progress_bars_disabled()
        disable_progress_bars()
        self._was_datasets_enabled = is_progress_bar_enabled()
        disable_progress_bar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from huggingface_hub.utils import enable_progress_bars
        from datasets.utils.logging import enable_progress_bar
        if not self._was_hub_disabled:
            enable_progress_bars()
        if self._was_datasets_enabled:
            enable_progress_bar()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Dataset manipulation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
from typing import Any
def filter_dataset(dataset: HFDataset, val: Any | list[Any], col_name: str):
    '''
    Efficiently filters a Hugging Face Dataset, returning rows where `dataset[col_name]` matches any
    value in `val`. Uses Arrow format with fast, vectorized filtering and batching.

    Filtering with `dataset.filter()` and numpy is slow for large datasets (see:
    https://github.com/huggingface/datasets/issues/1796). On the other hand, directly filtering
    with pyarrow is much faster, but produces incorrect row ordering if the dataset has been
    modified with dataset.select() or dataset.shuffle(). This is because such operations change
    the internal indices, and direct filtering with pyarrow ignores these.

    Approach:
    This function combines fast, vectorized PyArrow filtering with batched `.filter()` calls on
    the dataset in Arrow format.
    - If the dataset is unmodified, filtering is nearly instantaneous and equivalent to native
      Arrow filtering (<1s per 1M rows).
    - If the dataset has been reordered (e.g., via `.select()` or `.shuffle()`), batched Arrow
      filtering is still much faster and safer than using numpy or pandas, and it preserves the
      correct row order (<7s per 1M rows).
    
    Args:
        dataset (HFDataset): Hugging Face Dataset to filter.
        val (Any | list[Any]): Value(s) to match in the column.
        col_name (str): Column name to filter by.

    Returns:
        HFDataset: Filtered dataset in the original format.
    '''
    import pyarrow as pa, pyarrow.compute as pc
    from contextlib import nullcontext

    # make sure that the query (val) is always a list (never a single value)
    val = np.asarray(val)
    val = val[..., np.newaxis] if val.shape == () else val
    val = pa.array(val)

    ctx = disable_hf_progress_bars if len(dataset) < 500_000 else nullcontext
    num_proc = None if len(dataset) < 500_000 else 8
    
    def fast_arrow_filter(batch):
        col_values = batch[col_name]
        mask = pc.is_in(col_values, val)
        return mask.to_pylist()
    
    with ctx():
        filtered_ds = dataset.with_format('arrow')
        filtered_ds = filtered_ds.filter(
            fast_arrow_filter, batched=True, batch_size=1000, num_proc=num_proc
        )
    
    # Revert back to the original dataset format
    filtered_ds = filtered_ds.with_format(dataset.format['type'])
    return filtered_ds


def sort_dataset_by_columns(dataset: HFDataset, sort_columns: list[str]):
    '''
    Sorts a Hugging Face Dataset by specified columns and returns a new dataset with rows in sorted order.
    Note: sort_columns is a list of columns to sort by, in order of priority.
    '''
    
    # Convert dataset to Pandas DataFrame
    df = dataset.to_pandas()

    # Sort DataFrame by specified columns
    df_sorted = df.sort_values(by=sort_columns)

    # Extract sorted IDs
    sorted_ids = df_sorted.index.tolist()

    # Select rows by sorted IDs
    sorted_dataset = dataset.select(sorted_ids)

    return sorted_dataset


def update_episode_indices(dataset: HFDataset):
    '''
    Resets `episode_index` to start from 0 and increases sequentially (e.g., 0, 1, 2,...).
    Uses `frame_index`  to detect episode boundaries (where 0 marks a new episode). 
    Useful after merging datasets where episode indices aren't sequential.
    '''

    # Initialize the new episode_index column
    new_episode_index = []
    current_episode = -1

    # Loop through the dataset and update the episode index based on frame_index
    for frame_idx in dataset['frame_index']:
        if frame_idx == 0: # detect start of a new episode
            current_episode += 1 # increment the episode counter
        new_episode_index.append(current_episode)
    
    # Override episode_index in the dataset
    dataset = dataset.remove_columns('episode_index')
    dataset = dataset.add_column('episode_index', new_episode_index)
    return dataset


def save_episode_video(
        dir_path: str, 
        dataset: HFDataset, 
        dataset_col: str, 
        episode_index=0, 
        extension='mp4',
        fps=30):
    '''
    Save a video file for a single episode from a specified dataset column.
    
    Example:
        ```python
        save_episode_video('logs/videos', dataset, 'observation.image', episode_index=2)
        # Output path: logs/videos/observation.image/episode_000002.mp4
        ```
    '''
    import imageio.v3 as iio

    episode_data = filter_dataset(dataset, episode_index, 'episode_index')
    frames = episode_data[dataset_col][:]

    file_name = f'episode_{episode_index:06d}.{extension}'
    out_path = Path(dir_path) / dataset_col / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One can use quality (compression) parameter in range [1-10], for example `quality=6`.
    # Default ~5. Better quality -> sharper image, and larger files.
    # However, it turns out it's better to keep the quality lower, and increase img resolution.
    # Then later, rescale the image to lower img resolution (when training).
    # This results in a better image quality, given the same img resolution.
    iio.imwrite(out_path, frames, fps=fps)
    return out_path


def save_episode_videos(
    dir_path: str,
    dataset: HFDataset,
    dataset_cols: list[str] = None,
    episode_ids: list[int] = None,
    extension='mp4',
    fps=30,
    show_progress=False,
):
    '''
    Save videos for multiple columns and episodes from a Hugging Face Dataset.

    Args:
        dir_path: Base directory where the video will be saved.
        dataset: Hugging Face Dataset containing episode data.
        dataset_cols: list of columns with image data. If None, auto-detects image columns.
        episode_ids: list of episode indices to save. If None, saves all episodes.
        extension: Video file extension (default: 'mp4').

    Returns:
        dict: Mapping from column name to list of saved video paths.

    Example:
    ```python
        save_episode_videos('logs/videos', dataset, episode_ids=[0, 2],
            dataset_cols=['observation.image.left', 'observation.image.right'],
        )
        # Output:
        {
          'observation.image.left': [
              Path('logs/videos/observation.image.left/episode_000000.mp4'),
              Path('logs/videos/observation.image.left/episode_000002.mp4')
          ],
          'observation.images.right': [
              Path('logs/videos/observation.images.right/episode_000000.mp4'),
              Path('logs/videos/observation.images.right/episode_000002.mp4')
          ]
        }
        ```
    '''
    if episode_ids is None:
        episode_ids = dataset.unique('episode_index')
    
    if dataset_cols is None:
        dataset_cols = [name for name, type in dataset.features.items() if isinstance(type, datasets.features.Image)]

    out_paths = {}
    for episode_index in tqdm(episode_ids, desc='Saving videos', disable=not show_progress):
        for col_name in dataset_cols:
            path = save_episode_video(
                dir_path, dataset, col_name, episode_index, extension, fps
            )
            out_paths.setdefault(col_name, []).append(path)
    return out_paths


import logging
class suppress_warnings:
    '''Suppress logging warnings (context manager or decorator).'''
    def __enter__(self):
        self.logger = logging.getLogger()
        self.prev_level = self.logger.level
        self.logger.setLevel(logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.prev_level)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Visualization
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import torch
from einops import rearrange

def to_channel_last(arr_like):
    arr_like = np.asarray(arr_like)
    if arr_like.shape[-1] != 3:
        arr_like = rearrange(arr_like, '... c h w -> ... h w c')
    return arr_like

def visualize_episode(source: Iterable | HFDataset | torch.utils.data.Dataset, episode_index: int, use_rerun=True, is_notebook=True):
    '''Visualize an episode.'''
    if isinstance(source, torch.utils.data.Dataset): 
        source = _to_hf_dataset(source, episode_index)
        _vis_hf_dataset(source, episode_index)
    elif isinstance(source, HFDataset): 
        _vis_hf_dataset(source, episode_index, use_rerun, is_notebook)
    elif isinstance(source, Iterable): 
        _vis_frames(source, use_rerun, is_notebook)
    else: raise TypeError(f'Unsupported source type: {type(source)}')

def _to_hf_dataset(dataset: torch.utils.data.Dataset, episode_index: int):
    start_i, end_i = dataset.episode_ranges.loc[episode_index]
    rows = [dataset[idx] for idx in range(start_i, end_i)]
    return datasets.Dataset.from_list(rows)

def _vis_hf_dataset(dataset: HFDataset, episode_index: int, use_rerun=True, is_notebook=True):
    assert type(dataset) == HFDataset
    episode_data = filter_dataset(dataset, episode_index, 'episode_index').with_format('np')
    
    if use_rerun:
        _vis_hf_episode(episode_data, episode_index, is_notebook)
    else:
        img_col_name = next((n for n, t in dataset.features.items() if 'image' in n), None)
        frames = episode_data[img_col_name][:]
        _vis_frames(frames, use_rerun, is_notebook)

def _vis_hf_episode(episode_data: HFDataset, episode_index: int, is_notebook: bool):
    '''Visualize an episode using Rerun.'''
    # We start a new recording_id each time, so visualizing the same episode multiple times
    # doesn't duplicate the data. To visualize more episodes at once, remove 
    # recording_id and add spawn.
    _setup_rerun(episode_index, is_notebook)

    for row in episode_data:
        frame_index = row.get('frame_index')
        if frame_index is not None: rr.set_time('frame_index', sequence=frame_index)

        timestamp = row.get('timestamp')
        if timestamp is not None: rr.set_time('seconds', duration=timestamp)

        for key in row.keys():
            if 'action' in key:
                rr.log(f'{key}/a', rr.Scalars(row[key]))
            
            elif 'observation.state' in key:
                rr.log(f'{key}/o', rr.Scalars(row[key]))

            elif 'image' in key:
                rr.log(key, rr.Image(to_channel_last(row[key])))

def _vis_frames(episode_frames: Iterable, use_rerun=True, is_notebook=True):
    episode_frames = to_channel_last(episode_frames)
    
    if use_rerun:
        _setup_rerun(-1, is_notebook)

        for i, frame in enumerate(episode_frames):
            rr.set_time('frame_index', sequence=i)
            rr.log("image", rr.Image(np.array(frame)))
    
    else:
        for i, frame in enumerate(episode_frames):
            sleep(0.005)
            clear_output(True)
            plt.imshow(frame)
            plt.title(f'step: {i}')
            plt.show()

def _setup_rerun(episode_index, is_notebook: bool):
    rr.init(f'episode_data_{episode_index}', recording_id=uuid.uuid4())
    if is_notebook:
        rr.notebook_show(width=1000, height=700)
    else:
        rr.spawn()


def plot_attn_heatmap(attn_mask, font_scale=0.7, figsize=(10,4), cmap='inferno'):
    '''
    Plot an attention mask heatmap. Pass 1D or 2D array-like attention/padding 
    mask. 1D array (usually a padding mask) is expanded to 2D.
    '''
    import seaborn as sb
    attn_mask = np.asarray(attn_mask)
    if attn_mask.ndim == 1: attn_mask = attn_mask[None, :]
    if attn_mask.shape[0] == attn_mask.shape[1]: figsize = (8,7)

    
    plt.figure(figsize=figsize)
    sb.set_context("notebook", font_scale=font_scale)
    ax = sb.heatmap(attn_mask, linewidths=0.1, cbar=True, cmap=cmap, linecolor='grey')
    ax.set_title('attention mask')
    ax.set_ylabel('Q')
    ax.set_xlabel('K')
    plt.show()
