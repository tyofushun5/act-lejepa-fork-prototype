# Adapted from: https://github.com/tonyzhaozh/act
# Copyright (c) 2021 Tony Zhao
# Licensed under the MIT License
# Modification of original ACT adapted to this codebase.

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers import ResNetModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, pack

from transformer_utils.original import DecoderLayer, EncoderLayer

class ActConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = -1,
        state_dim: int = -1,
        encoder: dict = None,
        decoder: dict = None,
        horizon: dict[str, int] = {},  # e.g., {'action': 10}
        **kwargs,
    ):
        self.encoder = PretrainedConfig(**encoder)
        self.decoder = PretrainedConfig(**decoder)

        self.encoder.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        super().__init__(**kwargs)

########################## ACT Model ##########################

class ActEncoder(PreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        # Embed tokens - robot state (joints) & images
        self.state_emb = nn.Linear(config.state_dim, config.hidden_size)
        
        # Embed images
        self.img_emb = ResNetModel.from_pretrained('microsoft/resnet-18')
        n_feature_maps = self.img_emb.config.hidden_sizes[-1]
        # 1x1 conv projects feature maps to model hidden size at each spatial location
        # acts as a per-pixel linear projection
        self.img_proj = nn.Conv2d(n_feature_maps, config.hidden_size, kernel_size=1)
        self.pos_emb = PositionalEmbedding(config.hidden_size)

        # Embed task index
        self.task_emb = nn.Embedding(50, config.hidden_size)

        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def _get_img_tokens(self, imgs: torch.Tensor):
        '''
        1. ResNet is used as a feature extractor for images.
        The classification head (avgpool and final linear) is removed.
        Only the last hidden state is used (B n_feature_maps H W).

        2. Project feature maps to match the model's hidden_size.

        3. Then, positional tokens of shape (C H W) are created (2d sinusoidal),
        and are added to the hidden states.

        4. Instead of using the default average pooling (B C), hidden states
        are flattened to preserve the spatial dimension (B H*W C).
        Flattening keeps information from each spatial location, while avgpool 
        aggregates all into a single vector. This enables the transformer to process
        each position separately, having more fine-grained details.
        '''
        x = self.img_emb(pixel_values=imgs).last_hidden_state # (B n_feature_maps H W)

        x = self.img_proj(x) # (B C H W)
        
        x = x + self.pos_emb(x) # (B C H W) + (C H W) -> (B C H W)
        
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x # (B T C)

    def forward(self, **inputs: dict):
        task_token = self.task_emb(inputs['task_index']) # (B, C)
        state_token = self.state_emb(inputs['observation.state']) # (B C)
        img_tokens = self._get_img_tokens(inputs[self._get_img_key(**inputs)]) # (B, T, C)
        hidden_states, _ = pack([task_token, state_token, img_tokens], 'b * c')

        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _get_img_key(self, **inputs):
        x = getattr(self, '_img_key', None)
        if x is not None:
            return x
        else:
            x = next(k for k in inputs if k.startswith('observation.image') and 'is_pad' not in k)
            self._img_key = x
            return x

class ActDecoder(PreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        T = config.horizon['action'] # action chunk size
        # Learnable position tokens (action tokens)
        self.pos_emb = nn.Parameter(torch.randn(T, config.hidden_size))

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, encoder_hidden_states: torch.Tensor, **inputs: dict):
        hidden_states = repeat(self.pos_emb, 't c -> b t c', b=encoder_hidden_states.shape[0])
        attention_mask = self.get_attention_mask(**inputs)

        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    
    def get_attention_mask(self, **inputs: dict):
        pad_mask = inputs.get('action_is_pad')
        if pad_mask is None: return

        attention_mask = ~pad_mask

        # Broadcast to (B, nh, T, T)
        B, T = attention_mask.shape
        nh = self.config.num_attention_heads
        attention_mask = repeat(attention_mask, 'B Tk -> B nh Tq Tk', nh=nh, Tq=T)
        
        if self.config.causal: attention_mask = self._add_causal_mask(attention_mask)
        return attention_mask

    def _add_causal_mask(self, attention_mask: torch.Tensor):
        T = attention_mask.shape[-1]
        causal_mask = torch.tril(torch.ones((T, T), device=attention_mask.device)).bool()
        full_mask = attention_mask & causal_mask
        return full_mask


class ActModel(PreTrainedModel):
    config_class = ActConfig
    main_input_name = 'observation.state'

    # NOTE: this isn't used yet
    def _init_weights(self, module: nn.Module):
        '''xavier init'''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __init__(self, config: ActConfig):
        super().__init__(config)
        self.encoder = ActEncoder(config.encoder)
        self.decoder = ActDecoder(config.decoder)
        self.action_head = nn.Linear(config.decoder.hidden_size, config.action_dim)
        self.reconstruction_loss = 0.

    def forward(self, return_loss=True, **inputs: dict):
        '''Predict action chunks given observations.'''
        encoder_hidden_states = self.encoder(**inputs) # (B Te C)
        decoder_hidden_states = self.decoder(encoder_hidden_states, **inputs) # (B action_chunk_size C)
        # pass through the action head
        action_pred = self.action_head(decoder_hidden_states) # (B chunk_size C)
        output = dict(action_pred=action_pred)

        # calculate loss
        labels = inputs.get('labels')
        if labels is not None:
            loss = self.loss_function(**(inputs | output))
            self.reconstruction_loss = loss
            output['loss'] = loss
        
        return ModelOutput(output)

    def loss_function(self, action_pred, labels, action_is_pad, **kwargs):
        '''Compute loss.'''
        assert action_pred.shape == labels.shape
        assert action_pred.shape[:-1] == action_is_pad.shape
        # Extract non-padded actions (both predicted and labels)
        action_pred = action_pred[~action_is_pad]
        labels = labels[~action_is_pad]
        return F.l1_loss(action_pred, labels)


########################## Policy ##########################
from collections import deque
from models.policy_wrapper import PolicyWrapper

class ActionChunkingPolicy(PolicyWrapper):
    @torch.inference_mode()
    def select_action(self, inputs: list[dict] | dict):
        '''
        From the paper: every k steps, the agent receives an observation, 
        generates the next k actions, and executes the actions in a sequence.

        The policy predicts a chunk of actions (of length k) when the action queue is empty.
        At each timestep, an action is popped from the queue and returned. This reduces 
        model calls by generating multiple actions at once and serving them sequentially.
        '''
        inputs = super().select_action(inputs)

        # If the action queue is empty, generate a new chunk of actions
        if len(self.history['action']) == 0:
            # Forward inputs through the model to get action chunk
            outputs = self.forward(**inputs)
            action_pred = outputs['action_pred'] # (B, n_action_steps, action_dim)

            # Denormalize actions to original scale
            action_pred = self.processor.denormalize({'action': action_pred})['action']

            # Store denormalized actions
            self._store(action_pred, 'action')

        # Pop and return the next denormalized action
        return self.history['action'].popleft() # (B, action_dim)

    def _store(self, data, key: str):
        data = rearrange(data, 'b t ... -> t b ...') # transpose B & T dim
        self.history[key].extend(data)

    def reset(self):
        '''
        Resets the action history queue to empty. 
        Uses a fixed maximum length (chunk_size).
        Stores shape: (chunk_size, B, action_dim)
        '''
        self.history = {
            'action': deque([], maxlen=self.model.config.horizon['action']) 
        }


########################## Helper - 2D Pos emb ##########################

class PositionalEmbedding(nn.Module):
    '''2D sinusoidal positional embedding.'''

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dim = hidden_size // 2
        self._2pi = 2 * 3.14
        self._eps = 1e-6

    def forward(self, x: torch.Tensor):
        mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_pos = mask.cumsum(1, dtype=torch.float32)
        x_pos = mask.cumsum(2, dtype=torch.float32)

        y_pos = y_pos / (y_pos[:, -1:, :] + self._eps) * self._2pi
        x_pos = x_pos / (x_pos[:, :, -1:] + self._eps) * self._2pi

        denominator = 10000 ** (2 * (torch.arange(self.dim, dtype=torch.float32, device=x.device) // 2) / self.dim)

        x_pos = x_pos.unsqueeze(-1) / denominator
        y_pos = y_pos.unsqueeze(-1) / denominator

        x_embed = torch.stack((x_pos[..., 0::2].sin(), x_pos[..., 1::2].cos()), dim=-1).flatten(3)
        y_embed = torch.stack((y_pos[..., 0::2].sin(), y_pos[..., 1::2].cos()), dim=-1).flatten(3)
        pos_emb = torch.cat((y_embed, x_embed), dim=3).permute(0, 3, 1, 2)
        return pos_emb
    
