# Normalize layers
# Should use it in most cases to avoid numerical instability and 
# obtain faster (stable) training.

import torch
from torch import nn
from typing import Literal


class StatsNormalize(nn.Module):
    '''
    Normalize specified keys in a batch using provided stats.

    Example:
    ```
        normalize = StatsNormalize(stats=dataset.stats, keys=['action'])
        batch = normalize(batch)
    ```
    '''
    def __init__(self, stats: dict, keys: list[str], normalize_type: Literal['mean_std', 'min_max'] = 'mean_std'):
        super().__init__()
        self.keys = keys
        self.normalize_type = normalize_type
        self.stats = stats

    def apply(self, batch: dict) -> dict:
        '''Apply normalization.'''
        batch = dict(batch) # shallow copy

        for key in batch:
            if key not in self.keys: continue
            x = batch[key]
            stats = self.get_stats(key, x.device)

            if self.normalize_type == 'mean_std':    
                batch[key] = (x - stats['mean']) / (stats['std'] + 1e-8)
            elif self.normalize_type == 'min_max':
                batch[key] = (x - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
            else:
                raise ValueError(f'Unknown normalization type: {self.normalize_type}')
        return batch

    def unapply(self, batch: dict) -> dict:
        '''Revert normalization.'''
        batch = dict(batch) # shallow copy

        for key in batch:
            if key not in self.keys: continue
            x = batch[key]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            stats = self.get_stats(key, x.device)

            if self.normalize_type == 'mean_std':
                batch[key] = x * (stats['std'] + 1e-8) + stats['mean']
            elif self.normalize_type == 'min_max':
                batch[key] = x * (stats['max'] - stats['min'] + 1e-8) + stats['min']
            else:
                raise ValueError(f'Unknown normalization type: {self.normalize_type}')
        return batch

    def forward(self, batch: dict):
        '''Apply normalization.'''
        return self.apply(batch)
    
    def get_stats(self, key, device=None):
        stats = {}
        for stat_name in ['mean', 'std', 'min', 'max']:
            stats[stat_name] = torch.as_tensor(self.stats[key][stat_name], dtype=torch.float32, device=device)
        return stats


if __name__ == '__main__':
    # Create a batch with a key 'action'
    batch = {'action': torch.tensor([[1.0, 2.0, 3.0], 
                                     [4.0, 5.0, 6.0]])}

    # Define dataset statistics for normalization
    stats = {
        'action': {
            'mean': torch.tensor([2.0, 3.0, 4.0]),
            'std': torch.tensor([1.0, 1.0, 1.0]),
            'min': torch.tensor([1.0, 2.0, 3.0]),
            'max': torch.tensor([3.0, 4.0, 5.0])
        }
    }

    # Initialize StatsNormalize
    normalize = StatsNormalize(stats=stats, keys=['action'], normalize_type='mean_std')

    # Apply normalization
    normalized_batch = normalize(batch)
    print('Normalized batch:', normalized_batch)

    # Apply unnormalization
    unnormalized_batch = normalize.unapply(normalized_batch)
    print('Unnormalized batch:', unnormalized_batch)

    # Check that the original batch and the unnormalized batch are the same
    print('Allclose:', torch.allclose(batch['action'], unnormalized_batch['action']))
