'''ACT-LEJEPA: gradient-trained JEPA variant regularized with SIGReg (LeJEPA).

Kept in its own module so `act_jepa.py` stays faithful to the upstream
ACT-JEPA implementation. Reference for the LeJEPA objective/architecture:
`samples/le-wm-main`.
'''
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .act_model_original import ActDecoder
from .act_jepa import ActJepaConfig, ContextEncoder, TargetEncoder, Predictor


class ActLejepaConfig(ActJepaConfig):
    def __init__(
        self,
        *args,
        sigreg: dict = None,
        loss_weights: dict = None,
        target_update: str = 'grad',
        **kwargs
    ):
        if target_update != 'grad':
            raise ValueError(f"ActLejepaConfig requires target_update='grad', got {target_update!r}")
        super().__init__(*args, **kwargs)
        self.sigreg = PretrainedConfig(**(sigreg or {}))
        self.loss_weights = PretrainedConfig(**(loss_weights or {}))
        self.target_update = target_update


class GradTargetEncoder(TargetEncoder):
    '''
    Gradient-trained target encoder that embeds each state independently
    (no attention across timesteps, no task/current-state tokens).

    This mirrors the LeJEPA reference (le-wm), where every frame is encoded
    separately by a shared encoder. Because the label at position t is a
    function of state_t only, the target cannot drift toward encoding
    context-visible information (state_0 / task), which would trivialize the
    prediction objective while still satisfying SIGReg.
    '''
    def forward(self, **inputs: dict):
        states = inputs['observation.state'] # (B, T, D)
        B = states.shape[0]
        state_tokens = self.state_emb(states) # (B, T, C)

        # encode every timestep independently: (B, T, C) -> (B*T, 1, C)
        hidden_states = rearrange(state_tokens, 'b t c -> (b t) 1 c')
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)

        return rearrange(hidden_states, '(b t) 1 c -> b t c', b=B)


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


class Lejepa(PreTrainedModel):
    '''
    ACT-LEJEPA JEPA variant: trains the target encoder directly by gradient.
    '''
    config_class = ActLejepaConfig
    main_input_name = 'observation.state'

    def __init__(self, config: ActLejepaConfig):
        super().__init__(config)
        self.context_encoder = ContextEncoder(config.encoder)

        self.target_encoder = GradTargetEncoder(config.target_encoder)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

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
            abstract_labels = self.target_encoder(**inputs) # (B T_target C)
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
    def copy_context_to_target_encoder(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())


class ActLejepaModel(PreTrainedModel):
    '''
    ACT-LEJEPA model: ACT action decoder on top of gradient-trained LEJEPA.
    '''
    config_class = ActLejepaConfig
    main_input_name = 'observation.state'

    def __init__(self, config: ActLejepaConfig):
        super().__init__(config)
        self.jepa = Lejepa(config)
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
    def copy_context_to_target_encoder(self):
        return self.jepa.copy_context_to_target_encoder()

    @property
    def encoder(self):
        return self.jepa.context_encoder
