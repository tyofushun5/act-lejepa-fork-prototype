from pathlib import Path
from .utils import download_data
from box import Box

class Metadata:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.info = Box.from_json(filename=self.dataset_path / 'info.json')
        self.stats = Box.from_json(filename=self.dataset_path / 'stats.json')

    @classmethod
    def from_hf(cls, repo_id: str, revision: str='main'):
        from .episode_dataset import DATASETS_HOME
        dataset_path = DATASETS_HOME / repo_id / 'meta'
        try:
            return cls(dataset_path)
        except (FileNotFoundError, NotADirectoryError):
            # Trigger download
            download_data(repo_id, revision, 'meta/', dataset_path.parent)
            return cls(dataset_path)