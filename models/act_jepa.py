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
        sigreg: dict = None,
        loss_weights: dict = None,
        target_update: str = 'grad',
        horizon: dict[str, int] = {},  # e.g., {'action': 10}
        **kwargs
    ):
        self.encoder = PretrainedConfig(**encoder)
        self.target_encoder = PretrainedConfig(**target_encoder)
        self.predictor = PretrainedConfig(**predictor)
        self.decoder = PretrainedConfig(**decoder)
        self.sigreg = PretrainedConfig(**(sigreg or {}))
        self.loss_weights = PretrainedConfig(**(loss_weights or {}))

        self.encoder.state_dim = state_dim
        self.target_encoder.state_dim = state_dim
        self.predictor.encoder_hidden_size = self.encoder.hidden_size
        self.decoder.action_dim = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.target_update = target_update
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

    def forward(self, **inputs: dict):
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
    

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, is_pad=None, batch_first=False):
        """
        Compute SIGReg over batch distributions at each time step.

        Args:
            proj: Latents shaped either (T, B, D), (B, T, D) when
                batch_first=True, or a single pooled set (N, D).
            is_pad: Optional bool mask for padded sequence elements. For
                batch_first=True this must be shaped (B, T).
            batch_first: Whether a 3D proj tensor is shaped (B, T, D).
        """
        proj = proj.to(dtype=torch.float32)
        if proj.ndim == 2:
            if is_pad is not None:
                raise ValueError("is_pad is only supported for 3D SIGReg inputs")
            proj = proj.unsqueeze(0)
            valid_mask = torch.ones(proj.shape[:2], dtype=torch.bool, device=proj.device)
        elif proj.ndim == 3:
            if batch_first:
                proj = proj.transpose(0, 1)

            if is_pad is None:
                valid_mask = torch.ones(proj.shape[:2], dtype=torch.bool, device=proj.device)
            else:
                is_pad = is_pad.to(device=proj.device, dtype=torch.bool)
                expected_shape = (proj.size(1), proj.size(0)) if batch_first else proj.shape[:2]
                if tuple(is_pad.shape) == tuple(expected_shape):
                    valid_mask = ~is_pad.transpose(0, 1) if batch_first else ~is_pad
                elif not batch_first and tuple(is_pad.shape) == (proj.size(1), proj.size(0)):
                    valid_mask = ~is_pad.transpose(0, 1)
                else:
                    raise ValueError(
                        f"is_pad shape {tuple(is_pad.shape)} is incompatible with "
                        f"proj shape {tuple(proj.shape)} and {batch_first=}"
                    )
        else:
            raise ValueError(f"SIGReg expects a 2D or 3D tensor, got shape {tuple(proj.shape)}")

        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (proj @ A).unsqueeze(-1) * self.t

        valid = valid_mask.to(dtype=proj.dtype).unsqueeze(-1).unsqueeze(-1)
        counts = valid_mask.sum(dim=1)
        counts_safe = counts.to(dtype=proj.dtype).clamp_min(1).view(-1, 1, 1)
        cos_mean = (x_t.cos() * valid).sum(dim=1) / counts_safe
        sin_mean = (x_t.sin() * valid).sum(dim=1) / counts_safe

        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * counts.to(dtype=proj.dtype).unsqueeze(-1)
        statistic = statistic[counts > 0]
        if statistic.numel() == 0:
            return proj.new_zeros(())
        return statistic.mean()


class Jepa(PreTrainedModel):
    '''
    JEPA model consists of a context encoder, target encoder, and a predictor.

    Target encoder update behavior is selected by config.target_update:
    "ema" for the original ACT-JEPA target and "grad" for ACT-LEJEPA.
    '''
    config_class = ActJepaConfig
    main_input_name = 'observation.state'

    def __init__(self, config: ActJepaConfig):
        super().__init__(config)
        self.context_encoder = ContextEncoder(config.encoder)

        self.target_encoder = TargetEncoder(config.target_encoder)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_update = getattr(config, 'target_update', 'grad')
        if self.target_update not in {'ema', 'grad'}:
            raise ValueError(f"Unsupported target_update: {self.target_update}")
        if self.target_update == 'ema':
            self.target_encoder.eval()
            self.target_encoder.requires_grad_(False)

        self.predictor = Predictor(config.predictor)
        self.abstract_loss_weight = getattr(config.loss_weights, 'abstract', 1.0)
        self.target_sigreg_weight = getattr(
            config.loss_weights,
            'target_sigreg',
            getattr(config.sigreg, 'target_weight', getattr(config.sigreg, 'weight', 0.0)),
        )
        self.context_sigreg_weight = getattr(
            config.loss_weights,
            'context_sigreg',
            getattr(config.sigreg, 'context_weight', 0.0),
        )
        self.sigreg = SIGReg(
            knots=getattr(config.sigreg, 'knots', 17),
            num_proj=getattr(config.sigreg, 'num_proj', 1024),
        )

        self.abstract_loss = 0.
        self.target_sigreg_loss = 0.
        self.context_sigreg_loss = 0.
        self.sigreg_loss = 0.
        self.weighted_sigreg_loss = 0.
        self.jepa_loss = 0.

    def forward(self, return_loss=True, **inputs):
        '''Predicts abstract representation of a sequence of states.'''
        encoder_hidden_states = self.context_encoder(**inputs) # (B T_context C)
        output = dict(encoder_hidden_states=encoder_hidden_states)

        abstract_pred = self.predictor(encoder_hidden_states, **inputs) # (B T_target C)
        output['abstract_pred'] = abstract_pred

        # calculate loss
        labels = inputs.get('labels')
        if labels is not None:
            abstract_labels = self.encode_target(**inputs) # (B T_target C)
            output['abstract_labels'] = abstract_labels

            abstract_loss = self.abstract_loss_function(**(inputs | output))
            target_sigreg_loss = self.target_sigreg_loss_function(**(inputs | output))
            context_sigreg_loss = self.context_sigreg_loss_function(**(inputs | output))
            sigreg_loss = target_sigreg_loss + context_sigreg_loss
            weighted_sigreg_loss = self.weighted_sigreg_loss_function(
                target_sigreg_loss,
                context_sigreg_loss,
            )
            loss = self.loss_function(abstract_loss, weighted_sigreg_loss)

            self.abstract_loss = abstract_loss
            self.target_sigreg_loss = target_sigreg_loss
            self.context_sigreg_loss = context_sigreg_loss
            self.sigreg_loss = sigreg_loss
            self.weighted_sigreg_loss = weighted_sigreg_loss
            self.jepa_loss = loss
            output['abstract_loss'] = abstract_loss
            output['target_sigreg_loss'] = target_sigreg_loss
            output['context_sigreg_loss'] = context_sigreg_loss
            output['sigreg_loss'] = sigreg_loss
            output['weighted_sigreg_loss'] = weighted_sigreg_loss
            output['jepa_loss'] = loss
            output['loss'] = loss
        
        return ModelOutput(output)

    def encode_target(self, **inputs):
        if self.target_update == 'grad':
            return self.target_encoder(**inputs)

        self.target_encoder.eval()
        with torch.no_grad():
            return self.target_encoder(**inputs)

    def abstract_loss_function(self, abstract_pred, abstract_labels, **kwargs):
        is_pad = kwargs['observation.state_is_pad']
        assert abstract_pred.shape == abstract_labels.shape
        assert abstract_pred.shape[:-1] == is_pad.shape
        # Extract non-padded tokens
        abstract_pred = abstract_pred[~is_pad]
        abstract_labels = abstract_labels[~is_pad]
        return F.l1_loss(abstract_pred, abstract_labels)

    def target_sigreg_loss_function(self, abstract_labels, **kwargs):
        if self.target_sigreg_weight <= 0:
            return abstract_labels.new_zeros(())

        is_pad = kwargs['observation.state_is_pad']
        assert abstract_labels.shape[:-1] == is_pad.shape
        return self.sigreg(abstract_labels, is_pad=is_pad, batch_first=True)

    def context_sigreg_loss_function(self, encoder_hidden_states, **kwargs):
        if self.context_sigreg_weight <= 0:
            return encoder_hidden_states.new_zeros(())

        return self.sigreg(encoder_hidden_states, batch_first=True)

    def weighted_sigreg_loss_function(self, target_sigreg_loss, context_sigreg_loss):
        return (
            self.target_sigreg_weight * target_sigreg_loss
            + self.context_sigreg_weight * context_sigreg_loss
        )

    def loss_function(self, abstract_loss, weighted_sigreg_loss):
        return self.abstract_loss_weight * abstract_loss + weighted_sigreg_loss
    
    @torch.no_grad()
    def update_target_encoder(self, m: float):
        '''
        EMA update used by baseline ACT-JEPA configs.
        '''
        assert 0 <= m <= 1, f'EMA momentum is not in the valid range [0, 1] {m=}'
        # NOTE: as m approaches 1.0 (later stage of training), the target encoder 
        # updates become extremely slow, almost freezing its parameters.
        for (name_c, param_c), (name_t, param_t) in zip(self.context_encoder.named_parameters(), self.target_encoder.named_parameters()):
            assert name_c == name_t, f'params names must be equal: {name_c=}, {name_t=}'
            param_t.data.mul_(m).add_((1.0 - m) * param_c.data)

    @torch.no_grad()
    def copy_context_to_target_encoder(self):
        '''
        Hard-sync helper kept for backward-compatible checkpoints/scripts.

        Unlike EMA, this copies the full state_dict, including non-parameter
        buffers such as normalization running statistics.
        '''
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        if self.target_update == 'ema':
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
        self.target_sigreg_loss = 0.
        self.context_sigreg_loss = 0.
        self.sigreg_loss = 0.
        self.weighted_sigreg_loss = 0.
        self.jepa_loss = 0.
        self.reconstruction_loss_weight = getattr(
            config.loss_weights,
            'reconstruction',
            getattr(config.loss_weights, 'action', 1.0),
        )
        self.jepa_loss_weight = getattr(config.loss_weights, 'jepa', 1.0)
    
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
            self.target_sigreg_loss = output.target_sigreg_loss
            self.context_sigreg_loss = output.context_sigreg_loss
            self.sigreg_loss = output.sigreg_loss
            self.weighted_sigreg_loss = output.weighted_sigreg_loss
            self.jepa_loss = output.jepa_loss

            loss = self.loss_function(self.reconstruction_loss, self.jepa_loss)
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
    
    def loss_function(self, reconstruction_loss, jepa_loss):
        '''Compute weighted action and JEPA losses.'''
        return self.reconstruction_loss_weight * reconstruction_loss + self.jepa_loss_weight * jepa_loss
    
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
