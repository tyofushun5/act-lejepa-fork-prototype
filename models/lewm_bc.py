from transformers import ViTConfig, ViTModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .act_model_original import ActDecoder


class LeWmBcConfig(PretrainedConfig):
    encoder_defaults = {
        'image_size': 224,
        'patch_size': 14,
        'hidden_size': 192,
        'intermediate_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 3,
        'hidden_act': 'gelu',
        'attention_dropout': 0.0,
        'hidden_dropout_prob': 0.0,
        'attention_probs_dropout_prob': 0.0,
    }
    predictor_defaults = {
        'hidden_size': 192,
        'output_dim': 192,
        'intermediate_size': 2048,
        'num_hidden_layers': 6,
        'num_attention_heads': 16,
        'dim_head': 64,
        'attention_dropout': 0.1,
        'emb_dropout': 0.0,
    }
    projector_defaults = {'hidden_size': 2048, 'norm': 'batch_norm'}
    sigreg_defaults = {'weight': 0.09, 'knots': 17, 'num_proj': 1024}
    loss_weight_defaults = {'action': 1.0, 'pred': 1.0}

    def __init__(
        self,
        action_dim: int = -1,
        state_dim: int = -1,
        encoder: dict = None,
        action_encoder: dict = None,
        predictor: dict = None,
        projector: dict = None,
        pred_proj: dict = None,
        decoder: dict = None,
        sigreg: dict = None,
        loss_weights: dict = None,
        history_size: int = 3,
        num_preds: int = 1,
        embed_dim: int = None,
        horizon: dict[str, int] = {},
        **kwargs,
    ):
        encoder = self.encoder_defaults | (encoder or {})
        predictor = self.predictor_defaults | (predictor or {})
        projector = self.projector_defaults | (projector or {})
        pred_proj = self.projector_defaults | (pred_proj or {})
        decoder_defaults = {
            k: encoder[k]
            for k in [
                'hidden_size',
                'intermediate_size',
                'num_attention_heads',
                'hidden_act',
                'attention_dropout',
            ]
        } | {'num_hidden_layers': 1, 'causal': False}
        decoder = decoder_defaults | (decoder or {})
        sigreg = self.sigreg_defaults | (sigreg or {})
        loss_weights = self.loss_weight_defaults | (loss_weights or {})

        self.encoder = PretrainedConfig(**encoder)
        self.embed_dim = embed_dim or self.encoder.hidden_size
        self.action_encoder = PretrainedConfig(**(action_encoder or {}))
        self.predictor = PretrainedConfig(**predictor)
        self.projector = PretrainedConfig(**projector)
        self.pred_proj = PretrainedConfig(**pred_proj)
        self.decoder = PretrainedConfig(**decoder)
        self.sigreg = PretrainedConfig(**sigreg)
        self.loss_weights = PretrainedConfig(**loss_weights)

        self.encoder.state_dim = state_dim
        self.action_encoder.input_dim = action_dim
        self.action_encoder.emb_dim = self.embed_dim
        self.predictor.input_dim = self.embed_dim
        self.predictor.hidden_size = getattr(self.predictor, 'hidden_size', self.encoder.hidden_size)
        self.predictor.output_dim = getattr(self.predictor, 'output_dim', self.predictor.hidden_size)
        self.decoder.action_dim = action_dim
        self.decoder.horizon = horizon

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.history_size = history_size
        self.num_preds = num_preds
        self.horizon = horizon
        super().__init__(**kwargs)


class LeWmObservationEncoder(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        vit_config = ViTConfig(
            image_size=getattr(config, 'image_size', 224),
            patch_size=getattr(config, 'patch_size', 14),
            num_channels=getattr(config, 'num_channels', 3),
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, 'hidden_act', 'gelu'),
            hidden_dropout_prob=getattr(config, 'hidden_dropout_prob', 0.0),
            attention_probs_dropout_prob=getattr(config, 'attention_probs_dropout_prob', 0.0),
            qkv_bias=getattr(config, 'qkv_bias', True),
            layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-12),
        )
        self.img_emb = ViTModel(vit_config, add_pooling_layer=False, use_mask_token=False)
        self.state_emb = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, state: torch.Tensor, image: torch.Tensor):
        state, image = self._ensure_sequence(state, image)
        B, T = state.shape[:2]

        flat_state = rearrange(state, 'b t d -> (b t) d').float()
        flat_image = rearrange(image, 'b t c h w -> (b t) c h w')
        img_emb = self.img_emb(
            pixel_values=flat_image,
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        state_emb = self.state_emb(flat_state)
        obs_emb = self.fusion(torch.cat([img_emb, state_emb], dim=-1))
        return rearrange(obs_emb, '(b t) c -> b t c', b=B, t=T)

    @staticmethod
    def _ensure_sequence(state: torch.Tensor, image: torch.Tensor):
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if image.ndim == 4:
            image = image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(f'expected image with 4 or 5 dims, got {tuple(image.shape)}')
        if state.ndim != 3:
            raise ValueError(f'expected state with 2 or 3 dims, got {tuple(state.shape)}')
        if image.shape[1] == 1 and state.shape[1] > 1:
            image = image.expand(-1, state.shape[1], -1, -1, -1)
        if state.shape[1] != image.shape[1]:
            raise ValueError(f'state/image horizon mismatch: {state.shape[1]} != {image.shape[1]}')
        return state, image


class LeWmEmbedder(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        smoothed_dim = getattr(config, 'smoothed_dim', config.input_dim)
        mlp_scale = getattr(config, 'mlp_scale', 4)
        self.patch_embed = nn.Conv1d(config.input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * config.emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * config.emb_dim, config.emb_dim),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


def _make_norm(name: str, hidden_dim: int):
    if name == 'batch_norm':
        return nn.BatchNorm1d(hidden_dim)
    if name == 'layer_norm':
        return nn.LayerNorm(hidden_dim)
    if name in {'none', None}:
        return nn.Identity()
    raise ValueError(f'unknown norm: {name!r}')


class LeWmMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, norm: str = 'batch_norm'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _make_norm(norm, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LeWmFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LeWmAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class LeWmConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = LeWmAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = LeWmFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LeWmTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.layers = nn.ModuleList([
            LeWmConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class LeWmARPredictor(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, config.num_frames, config.input_dim))
        self.dropout = nn.Dropout(getattr(config, 'emb_dropout', 0.0))
        self.transformer = LeWmTransformer(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_size,
            output_dim=config.output_dim,
            depth=config.num_hidden_layers,
            heads=config.num_attention_heads,
            dim_head=getattr(config, 'dim_head', 64),
            mlp_dim=config.intermediate_size,
            dropout=getattr(config, 'attention_dropout', 0.0),
        )

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=t.device)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj, is_pad=None, batch_first=False):
        if proj.ndim != 3:
            raise ValueError(f'SIGReg expects a 3D tensor, got {tuple(proj.shape)}')
        if batch_first:
            proj = proj.transpose(0, 1)
            if is_pad is not None:
                is_pad = is_pad.transpose(0, 1)

        proj = proj.to(dtype=torch.float32)
        if is_pad is None:
            valid_mask = torch.ones(proj.shape[:2], dtype=torch.bool, device=proj.device)
        else:
            valid_mask = ~is_pad.to(device=proj.device, dtype=torch.bool)

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


class LeWmBcModel(PreTrainedModel):
    config_class = LeWmBcConfig
    main_input_name = 'observation.state'
    all_tied_weights_keys = {}

    def __init__(self, config: LeWmBcConfig):
        super().__init__(config)
        self.obs_encoder = LeWmObservationEncoder(config.encoder)
        projector_hidden = getattr(config.projector, 'hidden_size', 2048)
        pred_proj_hidden = getattr(config.pred_proj, 'hidden_size', 2048)
        projector_norm = getattr(config.projector, 'norm', 'batch_norm')
        pred_proj_norm = getattr(config.pred_proj, 'norm', 'batch_norm')
        self.projector = LeWmMLP(
            config.encoder.hidden_size,
            projector_hidden,
            config.embed_dim,
            norm=projector_norm,
        )
        self.action_encoder = LeWmEmbedder(config.action_encoder)
        config.predictor.num_frames = config.history_size
        self.predictor = LeWmARPredictor(config.predictor)
        self.pred_proj = LeWmMLP(
            config.predictor.output_dim,
            pred_proj_hidden,
            config.embed_dim,
            norm=pred_proj_norm,
        )
        self.decoder = ActDecoder(config.decoder)
        self.action_head = nn.Linear(config.decoder.hidden_size, config.action_dim)
        self.sigreg = SIGReg(
            knots=getattr(config.sigreg, 'knots', 17),
            num_proj=getattr(config.sigreg, 'num_proj', 1024),
        )

        self.reconstruction_loss = 0.
        self.pred_loss = 0.
        self.sigreg_loss = 0.
        self.wm_loss = 0.

        self.action_loss_weight = getattr(config.loss_weights, 'action', 1.0)
        self.pred_loss_weight = getattr(config.loss_weights, 'pred', 1.0)
        self.sigreg_loss_weight = getattr(
            config.loss_weights,
            'sigreg',
            getattr(config.sigreg, 'weight', 0.0),
        )

    def forward(self, return_loss=True, **inputs):
        state = inputs['observation.state']
        image = inputs[self._get_img_key(**inputs)]
        emb = self.encode_observations(state, image)
        output = dict(emb=emb)

        decoder_hidden_states = self.decoder(emb[:, :1], **inputs)
        action_pred = self.action_head(decoder_hidden_states)
        output['action_pred'] = action_pred

        labels = inputs.get('labels')
        if labels is not None:
            reconstruction_loss = self.reconstruction_loss_function(**(inputs | output))
            pred_loss = self.prediction_loss_function(emb, **inputs)
            sigreg_loss = self.sigreg_loss_function(emb, **inputs)
            wm_loss = pred_loss + self.sigreg_loss_weight * sigreg_loss
            loss = (
                self.action_loss_weight * reconstruction_loss
                + self.pred_loss_weight * pred_loss
                + self.sigreg_loss_weight * sigreg_loss
            )

            self.reconstruction_loss = reconstruction_loss
            self.pred_loss = pred_loss
            self.sigreg_loss = sigreg_loss
            self.wm_loss = wm_loss
            output['reconstruction_loss'] = reconstruction_loss
            output['pred_loss'] = pred_loss
            output['sigreg_loss'] = sigreg_loss
            output['wm_loss'] = wm_loss
            output['loss'] = loss

        return ModelOutput(output)

    def encode_observations(self, state, image):
        hidden = self.obs_encoder(state, image)
        return self._apply_sequence_module(self.projector, hidden)

    def prediction_loss_function(self, emb, **inputs):
        actions = inputs.get('action')
        if actions is None or actions.ndim != 3:
            return emb.new_zeros(())
        ctx_len = min(
            int(self.config.history_size),
            emb.shape[1] - int(self.config.num_preds),
            actions.shape[1],
        )
        if ctx_len <= 0:
            return emb.new_zeros(())

        ctx_emb = emb[:, :ctx_len]
        ctx_act = actions[:, :ctx_len]
        act_emb = self.action_encoder(ctx_act)
        pred = self.predictor(ctx_emb, act_emb)
        pred = self._apply_sequence_module(self.pred_proj, pred)
        target = emb[:, self.config.num_preds:self.config.num_preds + ctx_len]

        valid = self._prediction_valid_mask(ctx_len, **inputs)
        per_step = (pred - target).pow(2).mean(dim=-1)
        if valid is not None:
            per_step = per_step[valid]
        if per_step.numel() == 0:
            return emb.new_zeros(())
        return per_step.mean()

    def sigreg_loss_function(self, emb, **inputs):
        is_pad = self._observation_pad_mask(emb.shape[1], **inputs)
        if is_pad is not None:
            is_pad = is_pad[:, :emb.shape[1]]
        return self.sigreg(emb, is_pad=is_pad, batch_first=True)

    def reconstruction_loss_function(self, action_pred, labels, action_is_pad, **kwargs):
        assert action_pred.shape == labels.shape
        assert action_pred.shape[:-1] == action_is_pad.shape
        valid = ~action_is_pad
        if not valid.any():
            return action_pred.new_zeros(())
        action_pred = action_pred[valid]
        labels = labels[valid]
        return F.l1_loss(action_pred, labels)

    def _prediction_valid_mask(self, ctx_len, **inputs):
        obs_pad = self._observation_pad_mask(self.config.num_preds + ctx_len, **inputs)
        action_pad = inputs.get('action_is_pad')
        if obs_pad is None and action_pad is None:
            return None
        valid = torch.ones(
            inputs['action'].shape[:2],
            dtype=torch.bool,
            device=inputs['action'].device,
        )[:, :ctx_len]
        if obs_pad is not None:
            context_valid = ~obs_pad[:, :ctx_len]
            target_valid = ~obs_pad[:, self.config.num_preds:self.config.num_preds + ctx_len]
            valid = valid & context_valid & target_valid
        if action_pad is not None:
            valid = valid & ~action_pad[:, :ctx_len]
        return valid

    def _observation_pad_mask(self, length, **inputs):
        masks = []
        for key in self._observation_pad_keys(**inputs):
            mask = inputs.get(key)
            if mask is not None:
                masks.append(mask[:, :length].to(dtype=torch.bool))
        if not masks:
            return None
        mask = masks[0]
        for other in masks[1:]:
            mask = mask | other
        return mask

    def _observation_pad_keys(self, **inputs):
        keys = ['observation.state_is_pad']
        img_key = getattr(self, '_img_key', None)
        if img_key is None:
            img_key = next((k for k in inputs if k.startswith('observation.image') and 'is_pad' not in k), None)
        if img_key is not None:
            keys.append(f'{img_key}_is_pad')
        return keys

    @staticmethod
    def _apply_sequence_module(module, x):
        B, T = x.shape[:2]
        y = module(rearrange(x, 'b t d -> (b t) d'))
        return rearrange(y, '(b t) d -> b t d', b=B, t=T)

    def _get_img_key(self, **inputs):
        x = getattr(self, '_img_key', None)
        if x is not None and x in inputs:
            return x
        x = next(k for k in inputs if k.startswith('observation.image') and 'is_pad' not in k)
        self._img_key = x
        return x

    @property
    def encoder(self):
        return self.obs_encoder
