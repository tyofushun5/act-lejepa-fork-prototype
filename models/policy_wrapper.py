import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

class ModelWrapper(nn.Module):
    '''Compose a model and delegate attribute access to it.

    Keeps the underlying model API (parameters, state_dict, to, etc.) available
    while allowing higher-level methods to be added.
    '''
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        '''
        Delegate everything to the model.
        '''
        try:
            # First, try to get the attribute from the parent class
            return super().__getattr__(name)
        except AttributeError:
            # If not found, delegate the request to the model
            return getattr(self.model, name)

class PolicyWrapper(ModelWrapper):
    '''Abstract Policy wrapper adding select_action & reset methods.'''

    def __init__(self, model: nn.Module | PreTrainedModel):
        super().__init__(model)
        self.reset()

    def reset(self):
        '''Resets any internal state of the policy.'''
        pass
    
    @torch.inference_mode()
    def select_action(self, inputs: list[dict] | dict) -> dict[str, torch.Tensor]:
        '''
        Selects and returns an action given input observations.
        Input: is a dictionary containing the current batch of observations.

        Returns: Predicted actions for the current timestep (B, action_dim).
        '''
        assert not self.training
        assert self.processor is not None

        # Safe collate
        inputs = inputs if isinstance(inputs, dict) else torch.utils.data.default_collate(inputs)
        return inputs
