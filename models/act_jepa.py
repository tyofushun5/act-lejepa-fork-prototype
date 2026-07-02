from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat, pack, unpack

from transformer_utils.original import DecoderLayer

from .act_model_original import ActEncoder, ActDecoder

class ActJepaConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = -1,
        state_dim: int = -1,
        encoder: dict = None,
        target_encoder: dict = None,
        predictor: dict = None,
        decoder: dict = None,
        horizon: dict[str, int] = {},  # e.g., {'action': 10}
        **kwargs
    ):
        self.encoder = PretrainedConfig(**encoder)
        self.target_encoder = PretrainedConfig(**target_encoder)
        self.predictor = PretrainedConfig(**predictor)
        self.decoder = PretrainedConfig(**decoder)

        self.encoder.state_dim = state_dim
        self.target_encoder.state_dim = state_dim
        self.predictor.encoder_hidden_size = self.encoder.hidden_size
        self.decoder.action_dim = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        super().__init__(**kwargs)


########################## ACT-JEPA Model ##########################

class ContextEncoder(ActEncoder):

    def forward(self, **inputs: dict):
        task_token = self.task_emb(inputs['task_index']) # (B, C)
        state_token = self._get_state_token(inputs['observation.state']) # (B, C)
        img_tokens = self._get_img_tokens(inputs[self._get_img_key(**inputs)]) # (B, T, C)
        hidden_states, _ = pack([task_token, state_token, img_tokens], 'b * c')

        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    
    def _get_state_token(self, states: torch.Tensor):
        state_token = states[:, 0, :] if len(states.shape) == 3 else states # (B, C)
        assert len(state_token.shape) == 2
        state_token = self.state_emb(state_token)
        return state_token


class TargetEncoder(ActEncoder):
    def __init__(self, config):
        super().__init__(config)
        T = config.horizon['observation.state']
        self.state_pos_emb = get_1d_sincos_pos_emb(T, config.hidden_size).to(self.device) # (T, C)

    @torch.no_grad()
    def forward(self, **inputs: dict):
        assert self.training == False
        task_token = self.task_emb(inputs['task_index']) # (B, C)
        state_tokens = self.state_emb(inputs['observation.state']) # (B, T, C)
        state_tokens = state_tokens + self.state_pos_emb.to(self.device)
        hidden_states, ps = pack([task_token, state_tokens], 'b * c')
        attention_mask = self.get_attention_mask(**inputs)

        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)

        task_token, state_tokens = unpack(hidden_states, ps, 'b * c')

        return state_tokens

    def get_attention_mask(self, **inputs):
        pad_mask = inputs.get('observation.state_is_pad') # (B, T)
        B = pad_mask.shape[0]
        task_is_pad = torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device)
        attention_mask = ~(pack([task_is_pad, pad_mask], 'b *')[0])

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


class Predictor(nn.Module):
    '''
    Predicts current and all future states.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.encoder_hidden_size, config.hidden_size)

        T = config.horizon['observation.state']
        # Positional tokens / latent
        self.pos_emb = nn.Parameter(torch.randn(T, config.hidden_size))

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        self.out_proj = nn.Linear(config.hidden_size, config.encoder_hidden_size)

    def forward(self, encoder_hidden_states: torch.Tensor, **inputs: dict):
        encoder_hidden_states = self.in_proj(encoder_hidden_states) # make narrow
        hidden_states = repeat(self.pos_emb, 't c -> b t c', b=encoder_hidden_states.shape[0])
        attention_mask = self.get_attention_mask(**inputs)
        
        # pass output through a transformer
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.out_proj(hidden_states) # map back to default hidden_size

        return hidden_states

    def get_attention_mask(self, **inputs):
        pad_mask = inputs.get('observation.state_is_pad') # (B, T)
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
    

class Jepa(PreTrainedModel):
    '''
    JEPA model consists of a context encoder, target encoder, and a predictor.

    Target encoder is a copy of a context encoder - weights are identical at initialization.
    Target encoder is always in eval mode and is updated through EMA not, gradient descent.
    '''
    config_class = ActJepaConfig
    main_input_name = 'observation.state'

    def __init__(self, config: ActJepaConfig):
        super().__init__(config)
        self.context_encoder = ContextEncoder(config.encoder)

        self.target_encoder = TargetEncoder(config.target_encoder)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(False)

        self.predictor = Predictor(config.predictor)

        self.abstract_loss = 0.

    def forward(self, return_loss=True, **inputs):
        '''Predicts abstract representation of a sequence of states.'''
        encoder_hidden_states = self.context_encoder(**inputs) # (B T_context C)
        output = dict(encoder_hidden_states=encoder_hidden_states)

        abstract_pred = self.predictor(encoder_hidden_states, **inputs) # (B T_target C)
        output['abstract_pred'] = abstract_pred

        # calculate loss
        labels = inputs.get('labels')
        if labels is not None:
            self.target_encoder.eval() # Put target encoder in eval
            with torch.no_grad():
                abstract_labels = self.target_encoder(**inputs) # (B T_target C)
                output['abstract_labels'] = abstract_labels

            loss = self.loss_function(**(inputs | output))
            self.abstract_loss = loss
            output['loss'] = loss
            output['abstract_loss'] = loss

        return ModelOutput(output)

    def loss_function(self, abstract_pred, abstract_labels, **kwargs):
        is_pad = kwargs['observation.state_is_pad']
        assert abstract_pred.shape == abstract_labels.shape
        assert abstract_pred.shape[:-1] == is_pad.shape
        # Extract non-padded tokens
        abstract_pred = abstract_pred[~is_pad]
        abstract_labels = abstract_labels[~is_pad]
        return F.l1_loss(abstract_pred, abstract_labels)

    @torch.no_grad()
    def update_target_encoder(self, m: float):
        '''
        Update the target encoder using EMA (not gradient descent).
        The target encoder slowly follows the context encoder.
        '''
        assert 0 <= m <= 1, f'EMA momentum is not in the valid range [0, 1] {m=}'
        # NOTE: as m approaches 1.0 (later stage of training), the target encoder
        # updates become extremely slow, almost freezing its parameters.
        for (name_c, param_c), (name_t, param_t) in zip(self.context_encoder.named_parameters(), self.target_encoder.named_parameters()):
            assert name_c == name_t, f'params names must be equal: {name_c=}, {name_t=}'
            param_t.data.mul_(m).add_((1.0 - m) * param_c.data)

    @torch.no_grad()
    def copy_context_to_target_encoder(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(False)


class ActJepaModel(PreTrainedModel):
    '''
    ACT-JEPA model consists of a context encoder, target encoder, predictor, and action decoder.
    '''
    config_class = ActJepaConfig
    main_input_name = 'observation.state'

    def __init__(self, config: ActJepaConfig):
        super().__init__(config)
        self.jepa = Jepa(config)
        self.decoder = ActDecoder(config.decoder)
        self.action_head = nn.Linear(config.decoder.hidden_size, config.action_dim)

        self.abstract_loss = 0.
        self.reconstruction_loss = 0.

    def forward(self, return_loss=True, **inputs):
        output = self.jepa.forward(return_loss, **inputs)
        encoder_hidden_states = output.encoder_hidden_states # (B Te C)
        decoder_hidden_states = self.decoder(encoder_hidden_states, **inputs) # (B action_chunk_size C)
        # pass through the action head
        action_pred = self.action_head(decoder_hidden_states) # (B action_chunk_size C)
        output['action_pred'] = action_pred

        # calculate loss
        labels = inputs.get('labels')
        if labels is not None:
            reconstruction_loss = self.reconstruction_loss_function(**(inputs | output))
            self.reconstruction_loss = reconstruction_loss
            output['reconstruction_loss'] = reconstruction_loss
            self.abstract_loss = output.abstract_loss

            loss = self.loss_function(self.reconstruction_loss, self.abstract_loss)
            output['loss'] = loss

        return output

    def reconstruction_loss_function(self, action_pred, labels, action_is_pad, **kwargs):
        '''Compute action reconstruction loss in the action space.'''
        assert action_pred.shape == labels.shape
        assert action_pred.shape[:-1] == action_is_pad.shape
        # Extract non-padded actions (both predicted and labels)
        action_pred = action_pred[~action_is_pad]
        labels = labels[~action_is_pad]
        return F.l1_loss(action_pred, labels)

    def loss_function(self, reconstruction_loss, abstract_loss):
        '''The total loss is calculated as action reconstruction loss + abstract loss.'''
        return reconstruction_loss + abstract_loss

    @torch.no_grad()
    def update_target_encoder(self, m: float):
        return self.jepa.update_target_encoder(m)

    @torch.no_grad()
    def copy_context_to_target_encoder(self):
        return self.jepa.copy_context_to_target_encoder()

    @property
    def encoder(self):
        return self.jepa.context_encoder


def get_1d_sincos_pos_emb(seq_len, hidden_size):
    '''Returns a [seq_len, hidden_size] tensor of sinusoidal positional embeddings.
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})
    where d_model = hidden_size, pos = position index, i = dimension index.
    '''
    pos = torch.arange(seq_len).unsqueeze(1) # (seq_len, 1)
    i = torch.arange(0, hidden_size, 2) # (hidden_size/2, )
    denominator = torch.pow(10000, i / hidden_size)
    pe = torch.zeros(seq_len, hidden_size)

    pe[:, 0::2] = torch.sin(pos / denominator)
    pe[:, 1::2] = torch.cos(pos / denominator)
    return pe
