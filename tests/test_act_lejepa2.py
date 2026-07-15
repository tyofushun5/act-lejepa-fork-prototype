from types import SimpleNamespace

import torch
from torch import nn
from transformers import ResNetModel
from transformers.configuration_utils import PretrainedConfig

from models.act_lejepa2 import (
    ActLejepa2Config,
    Lejepa2,
    ObservationEncoder,
    TokenType,
)


class _FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_sizes=[4])
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.conv(pixel_values))


class _MixingTrunk(nn.Module):
    """Small trunk that records calls and exposes cross-token dependencies."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, hidden_states, attention_mask=None):
        self.calls.append(
            {
                'hidden_shape': tuple(hidden_states.shape),
                'attention_mask': (
                    None if attention_mask is None else attention_mask.detach().clone()
                ),
                'training': self.training,
            }
        )

        if attention_mask is None:
            visible = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            visible = attention_mask[:, 0, 0, :]
        visible = visible.unsqueeze(-1)
        pooled = (hidden_states * visible).sum(dim=1, keepdim=True)
        pooled = pooled / visible.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden_states + self.mix * pooled


def _encoder_config():
    return PretrainedConfig(
        state_dim=2,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_act='relu',
        attention_dropout=0.1,
    )


def _make_encoder(monkeypatch, *, horizon=3, target_causal=False):
    monkeypatch.setattr(
        ResNetModel,
        'from_pretrained',
        lambda *args, **kwargs: _FakeResNet(),
    )
    encoder = ObservationEncoder(
        _encoder_config(),
        horizon={'observation.state': horizon},
        target_causal=target_causal,
    )
    encoder.shared_trunk = _MixingTrunk()
    return encoder


def _inputs(*, batch_size=2, horizon=3):
    return {
        'observation.image': torch.randn(batch_size, 3, 4, 4),
        'observation.state': torch.randn(batch_size, horizon, 2),
        'observation.state_is_pad': torch.zeros(batch_size, horizon, dtype=torch.bool),
        'task_index': torch.arange(batch_size),
    }


def test_context_and_target_are_two_calls_to_one_encoder(monkeypatch):
    torch.manual_seed(0)
    encoder = _make_encoder(monkeypatch)
    inputs = _inputs()
    type_id_calls = []
    encoder.token_type_emb.register_forward_pre_hook(
        lambda _module, args: type_id_calls.append(args[0].detach().clone())
    )

    context_state = encoder._get_state_tokens(
        inputs['observation.state'],
        context_only=True,
    )
    target_states = encoder._get_state_tokens(inputs['observation.state'])
    assert torch.equal(context_state, target_states[:, :1, :])

    encoder.train()
    context = encoder.encode_context(**inputs)
    target = encoder.encode_target(**inputs)

    assert context.shape == (2, 18, 8)
    assert target.shape == (2, 3, 8)
    assert [call['training'] for call in encoder.shared_trunk.calls] == [True, True]
    assert torch.equal(
        type_id_calls[0],
        torch.tensor(
            [[TokenType.TASK, TokenType.STATE] + [TokenType.IMAGE] * 16] * 2
        ),
    )
    assert torch.equal(
        type_id_calls[1],
        torch.tensor([[TokenType.TASK] + [TokenType.STATE] * 3] * 2),
    )

    encoder.eval()
    encoder.encode_context(**inputs)
    encoder.encode_target(**inputs)
    assert [call['training'] for call in encoder.shared_trunk.calls[-2:]] == [False, False]


def test_target_preserves_full_trajectory_mask_and_shared_gradients(monkeypatch):
    torch.manual_seed(1)
    encoder = _make_encoder(monkeypatch)
    inputs = _inputs()
    inputs['observation.state_is_pad'][1, -1] = True

    output = encoder.encode_target(**inputs)
    call = encoder.shared_trunk.calls[-1]
    attention_mask = call['attention_mask']

    assert output.shape == (2, 3, 8)
    assert call['hidden_shape'] == (2, 4, 8)
    assert attention_mask.shape == (2, 2, 4, 4)
    assert torch.equal(
        attention_mask[:, 0, 0, :],
        torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
    )

    output.sum().backward()
    assert encoder.state_emb.weight.grad is not None
    assert encoder.task_emb.weight.grad is not None
    assert encoder.shared_trunk.mix.grad is not None


def test_target_states_can_interact_across_time(monkeypatch):
    torch.manual_seed(2)
    encoder = _make_encoder(monkeypatch)
    inputs = _inputs(batch_size=1)
    inputs['observation.state'].zero_()

    baseline = encoder.encode_target(**inputs)
    changed_states = inputs['observation.state'].clone()
    changed_states[:, -1, :] = 1.0
    changed = encoder.encode_target(
        **(inputs | {'observation.state': changed_states})
    )

    assert not torch.allclose(baseline[:, 0, :], changed[:, 0, :])


def test_target_causal_mask_covers_task_and_full_state_sequence(monkeypatch):
    encoder = _make_encoder(monkeypatch, target_causal=True)
    encoder.encode_target(**_inputs(batch_size=1))

    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    attention_mask = encoder.shared_trunk.calls[-1]['attention_mask']
    assert torch.equal(attention_mask[0, 0], expected)


def test_lejepa2_registers_exactly_one_observation_encoder(monkeypatch):
    monkeypatch.setattr(
        ResNetModel,
        'from_pretrained',
        lambda *args, **kwargs: _FakeResNet(),
    )
    horizon = {
        'observation.image': 1,
        'observation.state': 3,
        'episode_index': 1,
        'task_index': 1,
        'action': 3,
    }
    common = {
        'hidden_size': 8,
        'intermediate_size': 16,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'hidden_act': 'relu',
        'attention_dropout': 0.0,
        'horizon': horizon,
    }
    config = ActLejepa2Config(
        action_dim=2,
        state_dim=2,
        encoder=common,
        predictor={
            **common,
            'hidden_size': 4,
            'intermediate_size': 8,
            'num_attention_heads': 1,
            'causal': False,
        },
        decoder=common | {'num_hidden_layers': 1, 'causal': False},
        horizon=horizon,
        sigreg={'weight': 0.0, 'knots': 5, 'num_proj': 8},
        target_update='grad',
        target_causal=False,
    )
    model = Lejepa2(config)
    output = model(
        **{
            'observation.image': torch.randn(2, 3, 8, 8),
            'observation.state': torch.randn(2, 3, 2),
            'observation.state_is_pad': torch.zeros(2, 3, dtype=torch.bool),
            'task_index': torch.tensor([0, 1]),
            'labels': torch.zeros(2, 3, 2),
        }
    )

    assert output.encoder_hidden_states.ndim == 3
    assert output.abstract_pred.shape == (2, 3, 8)
    assert output.abstract_labels.shape == (2, 3, 8)
    assert [module for module in model.modules() if isinstance(module, ObservationEncoder)] == [
        model.encoder
    ]
    assert not hasattr(model, 'context_encoder')
    assert not hasattr(model, 'target_encoder')
    assert not hasattr(model.encoder, 'context_norm')
    assert not hasattr(model.encoder, 'target_norm')
    assert model.encoder.token_type_emb.num_embeddings == TokenType.NUM_TYPES == 3

    tokenizer_names = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(('state_emb.weight', 'task_emb.weight'))
    }
    assert tokenizer_names == {
        'encoder.state_emb.weight',
        'encoder.task_emb.weight',
    }

    output.loss.backward()
    assert model.encoder.state_emb.weight.grad is not None
    assert model.encoder.task_emb.weight.grad is not None
    assert (
        model.encoder.shared_trunk.layers[0]
        .self_attn.attention_interface.in_proj_weight.grad
        is not None
    )
