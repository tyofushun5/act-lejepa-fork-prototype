import numpy as np

class FeatureConfig:
    '''
    Configuration for extracting features at specified time offsets (horizons).

    Example:
    ```
        # Extract 'obs' and 'action' at t-1, t, t+1
        cfg = FeatureConfig(name='action', horizon=[-1, 0, 1])
    ```
    '''
    def __init__(self, name: str, horizon: list[int]):
        self.name = name
        self.horizon = np.asarray(horizon)

    def __repr__(self):
        return f'FeatureConfig(name={self.name}, horizon={list(self.horizon)})'
    


def download_data(repo_id: str, revision: str, allow_patterns: str|list[str], dataset_path: str):
    '''
    Download the data for the specified data pattern, if not already downloaded.
    '''
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=dataset_path,
        max_workers=2
    )


def get_missing_values(container_values, required_values):
    '''Return which required_values are missing from container_values.

        container_values: Iterable of available values.
        required_values: Iterable of values that must be present.

    Returns:
        set: Missing values (empty set if none missing).
    '''
    container_set = set(container_values)
    required_set = set(required_values)
    return required_set - container_set
