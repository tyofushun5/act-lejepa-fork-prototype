from transformers import ResNetModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from transformer_utils.original import DecoderLayer, EncoderLayer

from .act_model_original import ActDecoder, PositionalEmbedding


class ActLejepa2Config(PretrainedConfig):
    has_no_defaults_at_init = True

    def __init__(
        self,
        action_dim: int = -1,
        state_dim: int = -1,
        encoder: dict = None,
        predictor: dict = None,
        decoder: dict = None,
        horizon: dict[str, int] = {},
        sigreg: dict = None,
        target_update: str = 'grad',
        use_token_type_embeddings: bool = True,
        target_causal: bool = False,
        **kwargs
    ):
        self.encoder = PretrainedConfig(**encoder)
        self.predictor = PretrainedConfig(**predictor)
        self.decoder = PretrainedConfig(**decoder)

        self.encoder.state_dim = state_dim
        self.predictor.encoder_hidden_size = self.encoder.hidden_size
        self.decoder.action_dim = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        super().__init__(**kwargs)

        self.sigreg = PretrainedConfig(**(sigreg or {}))
        self.target_update = target_update
        self.use_token_type_embeddings = use_token_type_embeddings
        self.target_causal = target_causal

        if target_update != 'grad':
            raise ValueError(f"ActLejepa2Config requires target_update='grad', got {target_update!r}")


class TokenType:
    TASK = 0
    STATE = 1
    IMAGE = 2
    NUM_TYPES = 3


class ObservationEncoder(nn.Module):
    """One encoder used for both context observations and target states."""

    def __init__(
        self,
        config: PretrainedConfig,
        horizon: dict[str, int],
        use_token_type_embeddings: bool = True,
        target_causal: bool = False,
    ):
        super().__init__()
        self.config = config
        self.target_causal = target_causal
        self.state_emb = nn.Linear(config.state_dim, config.hidden_size)
        self.img_emb = ResNetModel.from_pretrained('microsoft/resnet-18')
        n_feature_maps = self.img_emb.config.hidden_sizes[-1]
        self.img_proj = nn.Conv2d(n_feature_maps, config.hidden_size, kernel_size=1)
        self.pos_emb = PositionalEmbedding(config.hidden_size)
        self.task_emb = nn.Embedding(50, config.hidden_size)
        self.shared_trunk = SharedEncoderTrunk(config)
        self.token_type_emb = (
            nn.Embedding(TokenType.NUM_TYPES, config.hidden_size)
            if use_token_type_embeddings
            else None
        )

        T = horizon['observation.state']
        self.register_buffer(
            'state_pos_emb',
            get_1d_sincos_pos_emb(T, config.hidden_size),
            persistent=False,
        )

    def forward(self, **inputs: dict):
        return self.encode_context(**inputs)

    def encode_context(self, **inputs: dict):
        task_token = self._get_task_token(inputs['task_index']).unsqueeze(1)
        state_tokens = self._get_state_tokens(inputs['observation.state'], context_only=True)
        img_tokens = self._get_img_tokens(inputs[self._get_img_key(**inputs)])

        hidden_states = torch.cat([task_token, state_tokens, img_tokens], dim=1)
        B = hidden_states.shape[0]
        type_ids = torch.cat(
            [
                torch.full((B, 1), TokenType.TASK, device=hidden_states.device),
                torch.full((B, 1), TokenType.STATE, device=hidden_states.device),
                torch.full((B, img_tokens.shape[1]), TokenType.IMAGE, device=hidden_states.device),
            ],
            dim=1,
        )
        hidden_states = self.add_token_type_embeddings(
            hidden_states,
            type_ids.to(dtype=torch.long),
        )
        return self.shared_trunk(hidden_states)

    def encode_target(self, **inputs: dict):
        states = inputs['observation.state']
        state_tokens = self._get_state_tokens(states)
        B, T, _ = state_tokens.shape
        task_token = self._get_task_token(inputs['task_index']).unsqueeze(1)

        # The input layout and mask differ from the context call, but tokenization,
        # token roles, transformer layers, normalization, dropout, and parameters
        # are exactly the same encoder invocation.
        hidden_states = torch.cat([task_token, state_tokens], dim=1)
        type_ids = torch.full(
            (B, T + 1),
            TokenType.STATE,
            dtype=torch.long,
            device=hidden_states.device,
        )
        type_ids[:, 0] = TokenType.TASK

        hidden_states = self.add_token_type_embeddings(hidden_states, type_ids)
        hidden_states = self.shared_trunk(
            hidden_states,
            self.get_target_attention_mask(B, T, **inputs),
        )
        return hidden_states[:, 1:, :]

    def add_token_type_embeddings(self, hidden_states, type_ids):
        if self.token_type_emb is None:
            return hidden_states
        return hidden_states + self.token_type_emb(type_ids)

    def _get_task_token(self, task_index: torch.Tensor):
        task_index = task_index.to(dtype=torch.long)
        if task_index.ndim > 1:
            task_index = task_index.squeeze(-1)
        return self.task_emb(task_index)

    def _get_state_tokens(self, states: torch.Tensor, context_only: bool = False):
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim != 3:
            raise ValueError(
                f"ObservationEncoder expects states shaped (B, T, C), got {tuple(states.shape)}"
            )
        if context_only:
            if states.shape[1] == 0:
                raise ValueError("ObservationEncoder requires at least one context state")
            states = states[:, :1, :]

        T = states.shape[1]
        if T > self.state_pos_emb.shape[0]:
            raise ValueError(
                f"ObservationEncoder received {T} states, but its configured horizon is "
                f"{self.state_pos_emb.shape[0]}"
            )

        state_tokens = self.state_emb(states)
        return state_tokens + self.state_pos_emb[:T].to(
            device=state_tokens.device,
            dtype=state_tokens.dtype,
        )

    def _get_img_tokens(self, imgs: torch.Tensor):
        if imgs.ndim == 5:
            if imgs.shape[1] != 1:
                raise ValueError(
                    f"ObservationEncoder expects one context image, got shape {tuple(imgs.shape)}"
                )
            imgs = imgs[:, 0]

        x = self.img_emb(pixel_values=imgs).last_hidden_state
        x = self.img_proj(x)
        x = x + self.pos_emb(x)
        return rearrange(x, 'b c h w -> b (h w) c')

    def _get_img_key(self, **inputs):
        x = getattr(self, '_img_key', None)
        if x is not None:
            return x
        x = next(k for k in inputs if k.startswith('observation.image') and 'is_pad' not in k)
        self._img_key = x
        return x

    def get_target_attention_mask(self, B: int, T: int, **inputs):
        pad_mask = inputs.get('observation.state_is_pad')
        if pad_mask is None:
            pad_mask = torch.zeros(
                (B, T),
                dtype=torch.bool,
                device=inputs['observation.state'].device,
            )
        else:
            pad_mask = pad_mask.to(
                device=inputs['observation.state'].device,
                dtype=torch.bool,
            )
            if pad_mask.ndim > 2:
                pad_mask = pad_mask.squeeze(-1)
            if pad_mask.ndim != 2 or pad_mask.shape[0] != B or pad_mask.shape[1] < T:
                raise ValueError(
                    f"ObservationEncoder expected a state padding mask compatible with "
                    f"(B, T)=({B}, {T}), got {tuple(pad_mask.shape)}"
                )
            pad_mask = pad_mask[:, :T]

        task_is_visible = torch.ones((B, 1), dtype=torch.bool, device=pad_mask.device)
        visible_tokens = torch.cat([task_is_visible, ~pad_mask], dim=1)

        nh = self.config.num_attention_heads
        attention_mask = repeat(
            visible_tokens,
            'B Tk -> B nh Tq Tk',
            nh=nh,
            Tq=T + 1,
        )
        if self.target_causal:
            causal_mask = torch.tril(
                torch.ones(
                    (T + 1, T + 1),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
            )
            attention_mask = attention_mask & causal_mask
        return attention_mask


class SharedEncoderTrunk(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.norm(hidden_states)


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
        weights = torch.full((knots,), 2 * dt, dtype=t.dtype, device=t.device)
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


class Lejepa2(PreTrainedModel):
    '''
    ACT-LEJEPA variant that invokes one observation encoder for both the
    context and target sequence layouts.
    '''
    config_class = ActLejepa2Config
    main_input_name = 'observation.state'

    def _init_weights(self, module):
        return

    def __init__(self, config: ActLejepa2Config):
        super().__init__(config)
        self.encoder = ObservationEncoder(
            config.encoder,
            horizon=config.horizon,
            use_token_type_embeddings=config.use_token_type_embeddings,
            target_causal=config.target_causal,
        )

        self.predictor = Predictor(config.predictor)
        self.sigreg_loss_weight = getattr(config.sigreg, 'weight', 0.0)
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
        self.post_init()

    def forward(self, return_loss=True, **inputs):
        encoder_hidden_states = self.encode_context(**inputs)
        output = dict(encoder_hidden_states=encoder_hidden_states)

        abstract_pred = self.predictor(encoder_hidden_states, **inputs)
        output['abstract_pred'] = abstract_pred

        labels = inputs.get('labels')
        if labels is not None:
            abstract_labels = self.encode_target(**inputs)
            output['abstract_labels'] = abstract_labels

            abstract_loss = self.abstract_loss_function(**(inputs | output))
            target_sigreg_loss = self.target_sigreg_loss_function(**(inputs | output))
            context_sigreg_loss = self.context_sigreg_loss_function(**(inputs | output))
            sigreg_loss = self.sigreg_loss_function(
                target_sigreg_loss,
                context_sigreg_loss,
            )
            weighted_sigreg_loss = self.weighted_sigreg_loss_function(sigreg_loss)
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

    def encode_context(self, **inputs):
        return self.encoder.encode_context(**inputs)

    def encode_target(self, **inputs):
        return self.encoder.encode_target(**inputs)

    def abstract_loss_function(self, abstract_pred, abstract_labels, **kwargs):
        is_pad = kwargs['observation.state_is_pad']
        assert abstract_pred.shape == abstract_labels.shape
        assert abstract_pred.shape[:-1] == is_pad.shape
        abstract_pred = abstract_pred[~is_pad]
        abstract_labels = abstract_labels[~is_pad]
        return F.l1_loss(abstract_pred, abstract_labels)

    def target_sigreg_loss_function(self, abstract_labels, **kwargs):
        if self.sigreg_loss_weight <= 0:
            return abstract_labels.new_zeros(())

        is_pad = kwargs['observation.state_is_pad']
        assert abstract_labels.shape[:-1] == is_pad.shape
        return self.sigreg(abstract_labels, is_pad=is_pad, batch_first=True)

    def context_sigreg_loss_function(self, encoder_hidden_states, **kwargs):
        if self.sigreg_loss_weight <= 0:
            return encoder_hidden_states.new_zeros(())

        encoder_hidden_states = encoder_hidden_states[:, 1:, :]
        return self.sigreg(encoder_hidden_states, batch_first=True)

    def sigreg_loss_function(self, target_sigreg_loss, context_sigreg_loss):
        return target_sigreg_loss + context_sigreg_loss

    def weighted_sigreg_loss_function(self, sigreg_loss):
        return self.sigreg_loss_weight * sigreg_loss

    def loss_function(self, abstract_loss, weighted_sigreg_loss):
        return abstract_loss + weighted_sigreg_loss


class ActLejepa2Model(PreTrainedModel):
    '''
    ACT-LEJEPA2 model: ACT action decoder on top of shared-trunk LEJEPA.
    '''
    config_class = ActLejepa2Config
    main_input_name = 'observation.state'

    def _init_weights(self, module):
        return

    def __init__(self, config: ActLejepa2Config):
        super().__init__(config)
        self.jepa = Lejepa2(config)
        self.decoder = ActDecoder(config.decoder)
        self.action_head = nn.Linear(config.decoder.hidden_size, config.action_dim)

        self.abstract_loss = 0.
        self.reconstruction_loss = 0.
        self.target_sigreg_loss = 0.
        self.context_sigreg_loss = 0.
        self.sigreg_loss = 0.
        self.weighted_sigreg_loss = 0.
        self.jepa_loss = 0.
        self.post_init()

    def forward(self, return_loss=True, **inputs):
        output = self.jepa.forward(return_loss, **inputs)
        encoder_hidden_states = output.encoder_hidden_states
        decoder_hidden_states = self.decoder(encoder_hidden_states, **inputs)
        action_pred = self.action_head(decoder_hidden_states)
        output['action_pred'] = action_pred

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
        assert action_pred.shape == labels.shape
        assert action_pred.shape[:-1] == action_is_pad.shape
        action_pred = action_pred[~action_is_pad]
        labels = labels[~action_is_pad]
        return F.l1_loss(action_pred, labels)

    def loss_function(self, reconstruction_loss, jepa_loss):
        return reconstruction_loss + jepa_loss

    @property
    def encoder(self):
        return self.jepa.encoder


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
