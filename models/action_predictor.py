from transformers.modeling_utils import PreTrainedModel

from .act_model_original import ActModel


class ActionPredictorModel(ActModel):
    '''
    Autoencoder (ACT), where the given encoder extracts features,
    and decoder maps them to actions.
    '''

    def __init__(self, config, encoder: PreTrainedModel):
        super().__init__(config)
        # this will be context encoder from JEPA, not ActEncoder
        self.encoder = encoder
    
    def forward(self, return_loss=True, **inputs):
        # Can be fine-tuned or linear probe (frozen backbone)
        if getattr(self.config, 'freeze_encoder', True):
            self.encoder.requires_grad_(False)
        else:
            self.encoder.requires_grad_(True)

        # Make sure the states are of shape (B, C)
        state_shape = inputs['observation.state'].shape
        if len(state_shape) == 3:
            inputs['observation.state'] = inputs['observation.state'][:, 0, :]

        return super().forward(return_loss, **inputs)
