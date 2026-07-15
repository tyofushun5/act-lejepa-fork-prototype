import torch
from torch import nn
from types import SimpleNamespace
from transformers import ResNetModel
from transformers.configuration_utils import PretrainedConfig

from models.act_lejepa2 import (
    ActLejepa2Config,
    Lejepa2,
    TargetEncoder,
    TokenType,
)


class _MixingTrunk(nn.Module):
    """Small shared trunk that makes cross-token dependencies observable."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(torch.tensor(1.0))
        self.last_hidden_shape = None
        self.last_attention_mask = None

    def forward(self, hidden_states, attention_mask=None):
        self.last_hidden_shape = tuple(hidden_states.shape)
        self.last_attention_mask = attention_mask.detach().clone()

        visible = attention_mask[:, 0, 0, :].unsqueeze(-1)
        pooled = (hidden_states * visible).sum(dim=1, keepdim=True)
        pooled = pooled / visible.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden_states + self.mix * pooled


class _SharedObservationEncoder(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.state_emb = nn.Linear(state_dim, hidden_size, bias=False)
        self.task_emb = nn.Embedding(8, hidden_size)
        self.shared_trunk = _MixingTrunk()
        self.last_type_ids = None

    def _get_task_token(self, task_index):
        return self.task_emb(task_index.to(dtype=torch.long))

    def add_token_type_embeddings(self, hidden_states, type_ids):
        self.last_type_ids = type_ids.detach().clone()
        return hidden_states


def _target_config(*, horizon=3, causal=False):
    return PretrainedConfig(
        state_dim=2,
        hidden_size=8,
        num_attention_heads=2,
        attention_dropout=0.0,
        causal=causal,
        horizon={'observation.state': horizon},
    )


def test_target_uses_shared_tokenizers_and_preserves_trajectory_sequence():
    torch.manual_seed(0)
    context = _SharedObservationEncoder(state_dim=2, hidden_size=8)
    target = TargetEncoder(_target_config(), branch_norm=False)
    states = torch.randn(2, 3, 2)
    pad_mask = torch.tensor([[False, False, False], [False, False, True]])

    output = target(
        context,
        target_trunk_eval=False,
        **{
            'observation.state': states,
            'observation.state_is_pad': pad_mask,
            'task_index': torch.tensor([1, 2]),
        },
    )

    assert output.shape == (2, 3, 8)
    assert not hasattr(target, 'state_emb')
    assert not hasattr(target, 'task_emb')
    assert context.shared_trunk.last_hidden_shape == (2, 4, 8)
    assert context.shared_trunk.last_attention_mask.shape == (2, 2, 4, 4)
    assert torch.equal(
        context.last_type_ids,
        torch.tensor(
            [
                [TokenType.TASK, TokenType.TARGET_STATE, TokenType.TARGET_STATE, TokenType.TARGET_STATE],
                [TokenType.TASK, TokenType.TARGET_STATE, TokenType.TARGET_STATE, TokenType.TARGET_STATE],
            ]
        ),
    )

    expected_visible_keys = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    assert torch.equal(
        context.shared_trunk.last_attention_mask[:, 0, 0, :],
        expected_visible_keys,
    )

    output.sum().backward()
    assert context.state_emb.weight.grad is not None
    assert context.task_emb.weight.grad is not None
    assert context.shared_trunk.mix.grad is not None


def test_target_states_can_interact_across_time():
    torch.manual_seed(1)
    context = _SharedObservationEncoder(state_dim=2, hidden_size=8)
    target = TargetEncoder(_target_config(), branch_norm=False)
    inputs = {
        'observation.state': torch.zeros(1, 3, 2),
        'observation.state_is_pad': torch.zeros(1, 3, dtype=torch.bool),
        'task_index': torch.tensor([0]),
    }

    baseline = target(context, target_trunk_eval=False, **inputs)
    changed_states = inputs['observation.state'].clone()
    changed_states[:, -1, :] = 1.0
    changed = target(
        context,
        target_trunk_eval=False,
        **(inputs | {'observation.state': changed_states}),
    )

    assert not torch.allclose(baseline[:, 0, :], changed[:, 0, :])


def test_target_causal_mask_covers_the_full_sequence():
    context = _SharedObservationEncoder(state_dim=2, hidden_size=8)
    target = TargetEncoder(_target_config(causal=True), branch_norm=False)

    target(
        context,
        target_trunk_eval=False,
        **{
            'observation.state': torch.zeros(1, 3, 2),
            'observation.state_is_pad': torch.zeros(1, 3, dtype=torch.bool),
            'task_index': torch.tensor([0]),
        },
    )

    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(context.shared_trunk.last_attention_mask[0, 0], expected)


def test_target_eval_pass_restores_shared_trunk_training_mode():
    context = _SharedObservationEncoder(state_dim=2, hidden_size=8)
    target = TargetEncoder(_target_config(), branch_norm=False)
    context.shared_trunk.train()

    target(
        context,
        target_trunk_eval=True,
        **{
            'observation.state': torch.zeros(1, 3, 2),
            'observation.state_is_pad': torch.zeros(1, 3, dtype=torch.bool),
            'task_index': torch.tensor([0]),
        },
    )

    assert context.shared_trunk.training


class _FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_sizes=[4])
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.conv(pixel_values))


def test_lejepa2_forward_uses_one_state_and_task_tokenizer(monkeypatch):
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
        target_encoder=common | {'causal': False},
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
        target_trunk_eval=True,
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
    tokenizer_names = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(('state_emb.weight', 'task_emb.weight'))
    }
    assert tokenizer_names == {
        'context_encoder.state_emb.weight',
        'context_encoder.task_emb.weight',
    }

    output.loss.backward()
    assert model.context_encoder.state_emb.weight.grad is not None
    assert model.context_encoder.task_emb.weight.grad is not None
    assert model.context_encoder.shared_trunk.layers[0].self_attn.attention_interface.in_proj_weight.grad is not None
