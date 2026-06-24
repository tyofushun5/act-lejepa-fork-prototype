from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange,  pack, unpack

from transformer_utils.original import EncoderLayer

from .policy_wrapper import PolicyWrapper
from collections import deque
from einops.layers.torch import Reduce
from copy import copy


class ARTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = -1,
        state_dim: int = -1,
        backbone_hidden_size=32,
        hidden_size = 128,
        num_hidden_layer=1,
        num_attention_heads=1,
        horizon: dict[str, int] = {},  # e.g., {'action': 10}
        **kwargs
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layer
        self.num_attention_heads = num_attention_heads
        self.horizon = horizon
        super().__init__(**kwargs)


class ARTransformerDecoder(PreTrainedModel):

    def __init__(self, config: ARTransformerConfig):
        super().__init__(config)

        self.img_emb = nn.Sequential(
            nn.Conv2d(3, config.backbone_hidden_size, kernel_size=8, stride=4, padding=1), nn.ReLU(),
            nn.Conv2d(config.backbone_hidden_size, 2*config.backbone_hidden_size, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2*config.backbone_hidden_size, config.hidden_size, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            Reduce('b c h w -> b c', reduction='mean'), # global avg pooling
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        self.act_emb = nn.Linear(config.action_dim, config.hidden_size)
        self.task_emb = nn.Embedding(50, config.hidden_size)

        # Learnable position tokens
        T = config.horizon['action'] + config.horizon['observation.state']
        self.pos_emb = nn.Parameter(torch.randn(T, config.hidden_size))

        # Even though we instantiate EncoderLayer, this is equivalent to the transformer decoder,
        # as we use causal_mask in the forward pass.
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def _get_img_tokens(self, imgs: torch.Tensor):
        '''
        Uses Conv NN as a feature extractor.
        '''
        B, T = imgs.shape[:2]
        imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
        imgs = self.img_emb(imgs)
        imgs = rearrange(imgs, '(b t) c -> b t c', b=B, t=T)
        return imgs
    

    def forward(self, **inputs: dict):
        B, T = inputs['observation.image'].shape[:2]

        # Embed images and actions
        img_tokens = self._get_img_tokens(inputs['observation.image']) # (B T C)
        act_tokens = self.act_emb(inputs['action']) # (B T C)
        task_token = self.task_emb(inputs['task_index']) # (B, C)

        # add positional emb per modality (as in DT, but different than RT1)
        pos_emb = self.pos_emb[:T]
        img_tokens += pos_emb
        act_tokens += pos_emb

        hidden_states, ps1 = pack([img_tokens, act_tokens], 'b t * c')
        # Interleave tokens. M is the number of tokens in a timestep.
        hidden_states = rearrange(hidden_states, 'b t m c -> b (t m) c')
        # add task token
        hidden_states, ps2 = pack([task_token, hidden_states], 'b * c')

        attention_mask = self.get_attention_mask(hidden_states)

        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        
        task_token, hidden_states = unpack(hidden_states, ps2, 'b * c') # remove task token
        hidden_states = rearrange(hidden_states, 'b (t m) c -> b t m c', t=T) # reverse interleave
        img_tokens, action_tokens = unpack(hidden_states, ps1, 'b t * c') # unpack
        return img_tokens, action_tokens

    def get_attention_mask(self, hidden_states):
        B, T, _ = hidden_states.shape
        nh = self.config.num_attention_heads
        attn_mask = torch.ones(B, nh, T, T, device=hidden_states.device).bool()
        if self.config.causal: attn_mask = self._add_causal_mask(attn_mask)
        return attn_mask

    def _add_causal_mask(self, attention_mask: torch.Tensor):
        T = attention_mask.shape[-1]
        causal_mask = torch.tril(torch.ones((T, T), device=attention_mask.device)).bool()
        full_mask = attention_mask & causal_mask
        return full_mask


class ARTransformerModel(PreTrainedModel):
    config_class = ARTransformerConfig

    def __init__(self, config: ARTransformerConfig):
        super().__init__(config)
        self.decoder = ARTransformerDecoder(config)
        self.action_head = nn.Linear(config.hidden_size, config.action_dim)
        self.reconstruction_loss = 0.
    
    def forward(self, return_loss=True, **inputs):
        '''Predicts the next action, given previous observations and actions.'''
        img_tokens, action_tokens = self.decoder(**inputs)

        # Pass img tokens (not action tokens!) through the action head
        # to predict the next action token (B num_actions C)
        action_pred = self.action_head(img_tokens)
        output = dict(action_pred=action_pred)

        # calculate loss
        labels = inputs.get('labels')
        if labels is not None:
            loss = self.loss_function(**(inputs | output))
            self.reconstruction_loss = loss
            output['loss'] = loss
        
        return ModelOutput(output)
    
    def loss_function(self, action_pred, labels, action_is_pad, **kwargs):
        assert action_pred.shape == labels.shape
        assert action_pred.shape[:-1] == action_is_pad.shape
        action_pred = action_pred[~action_is_pad]
        labels = labels[~action_is_pad]
        return F.mse_loss(action_pred, labels)


def disable_labels_name(func):
    '''
    Decorator that temporarily sets self.processor.labels_name to None during method execution.
    We use this during rollout to prevent loss calculation.
    '''
    def wrapper(self, *args, **kwargs):
        labels_name = self.processor.labels_name
        self.processor.labels_name = None
        try:
            return func(self, *args, **kwargs)
        finally:
            self.processor.labels_name = labels_name
    return wrapper

class ARTransformerPolicy(PolicyWrapper):

    @torch.inference_mode()
    @disable_labels_name
    def select_action(self, inputs: list[dict] | dict):
        '''
        Autoregressive Transformer policy for continuous-action behavior cloning.

        Causal Transformer policy that predicts next continuous action a_t given past
        (observations, actions). Trained with L2 loss. Similar to Decision Transformer architecture (minus Rtgs).

        This policy uses a sliding context window (stored in deques) to keep 
        track of past observations and actions for auto-regressive next-action prediction.
        '''
        inputs = super().select_action(inputs)

        # Store the observation (will also slide the context window)
        self._store(inputs['observation.image'], 'observation.image')

        inputs = self._prepare_inputs(inputs)

        # Forward inputs through the model to get action chunk
        outputs = self.forward(**inputs)
        action_pred = outputs['action_pred'] # (B, n_action_steps, action_dim)

        # Denormalize actions
        action_pred = self.processor.denormalize({'action': action_pred})['action']

        # Pluck out the last predicted action
        action_pred = action_pred[:, -1, :] # (B, action_dim)
        # Store denormalized action
        self._store(action_pred, 'action')
        return action_pred
    
    def _store(self, data, key: str):
        data = data.unsqueeze(1) # add fake T dim
        data = rearrange(data, 'b t ... -> t b ...') # transpose B & T dim
        self.history[key].extend(data)
    
    def reset(self):
        '''
        Resets the action history and observation history to empty.
        Uses a fixed maximum length (chunk_size).
        Stores shape: (chunk_size, B, action_dim)
        '''

        self.history = {
            'action': deque([], maxlen=self.model.config.horizon['action']),
            'observation.image': deque([], maxlen=self.model.config.horizon['observation.image'])
        }

    def _deque_to_tensor(self, data: deque):
        return rearrange(list(data), 't b ... -> b t ...')

    def _prepare_inputs(self, inputs: dict):
        '''Adds previous observations and actions to the inputs.'''
        inputs = dict(inputs)

        inputs['observation.image'] = self.history['observation.image']
        inputs['observation.image'] = self._deque_to_tensor(inputs['observation.image'])
        
        # Append dummy action at the very end.
        dummy_action = torch.zeros(
            (inputs['observation.image'].shape[0], self.model.config.action_dim), # (B, action_dim)
            device=inputs['observation.image'].device
        )
        # We shallow copy the deque, so we don't modify self.history
        inputs['action'] = copy(self.history['action'])
        inputs['action'].append(dummy_action)
        inputs['action'] = self._deque_to_tensor(inputs['action'])
        return inputs
