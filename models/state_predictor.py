from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from .act_jepa import Predictor

class StatePredictorConfig(PretrainedConfig):
    def __init__(
        self,
        state_dim: int = -1,
        predictor: dict = None,
        horizon: dict[str, int] = {},  # e.g., {'action': 10}
        **kwargs
    ):
        self.predictor = PretrainedConfig(**predictor)
        self.state_dim = state_dim
        self.horizon = horizon
        super().__init__(**kwargs)


class StatePredictorModel(nn.Module):
    config_class = StatePredictorConfig

    def __init__(self, config: StatePredictorConfig, encoder: PreTrainedModel):
        super().__init__()
        encoder_hidden_size = encoder.config.hidden_size
        config.predictor.encoder_hidden_size = encoder_hidden_size
        
        self.encoder = encoder
        self.predictor = Predictor(config.predictor)
        self.state_head = nn.Linear(encoder_hidden_size, config.state_dim)

        self.rmse_loss = self.ate_loss = 0.
    
    def forward(self, return_loss=True, **inputs):
        # Keep the full target sequence in inputs. The JEPA context encoder consumes
        # only the current state token internally (see ContextEncoder._get_state_token), 
        # while the predictor uses the full sequence masks/horizon for state prediction.
        assert len(inputs['observation.state'].shape) == 3

        # Pass through the frozen backbone
        self.encoder.eval()
        with torch.no_grad():
            encoder_hidden_states = self.encoder(**inputs) # (B, T, C)

        predictor_hidden_states = self.predictor(encoder_hidden_states, **inputs) # (B, T, C)
        pred = self.state_head(predictor_hidden_states) # (B, T, state_dim)
        output = dict(pred=pred)

        labels = inputs.get('labels')
        if labels is not None:
            output['loss'] = self.loss_function(**(inputs | output))
            output['rmse'] = self.compute_rmse(**(inputs | output))
            output['ate']  = self.compute_ate(**(inputs | output))
            self.rmse_loss = output['rmse']
            self.ate_loss = output['ate']

        return ModelOutput(output)
    
    @classmethod
    def loss_function(cls, pred, labels, **kwargs):
        '''Return MSE loss'''
        is_pad = kwargs.get('observation.state_is_pad')
        assert pred.shape == labels.shape
        assert pred.shape[:-1] == is_pad.shape
        pred = pred[~is_pad]
        labels = labels[~is_pad]
        return F.mse_loss(pred, labels)

    @classmethod
    def compute_rmse(cls, pred, labels, **kwargs):
        return cls.loss_function(pred, labels, **kwargs).sqrt()
    
    @classmethod
    def compute_ate(cls, pred, labels, **kwargs):
        '''
        Compute Absolute Trajectory Error (ATE) and account for padded tokens. 
        Expected shape: (B, T, C).
        
        ATE_t = ||p_t_pred - p_t_gt||_2 
              = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2 + (z_pred - z_gt)^2)
            
        Where (e.g., 3 dim proprioceptive state):
        - p_t_pred = predicted state at time t (x, y, z)
        - p_t_gt = ground truth state at time t (x, y, z)
        - x, y, z are the 3D position components
        
        The mean ATE is computed as the average ATE over all time steps.
        '''
        is_pad = kwargs.get('observation.state_is_pad')
        assert pred.shape == labels.shape and pred.shape[:2] == is_pad.shape
        pred = pred[~is_pad]
        labels = labels[~is_pad]

        # Compute the Euclidean distance (L2 norm) between the predicted and ground truth trajectories
        # Compute ATE for each time step (over C dim)
        ate = torch.norm(pred - labels, dim=-1)

        # Compute the mean ATE over all time steps
        mean_ate = torch.mean(ate)
        return mean_ate
