from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.training import Config
import models
from models import ActJepaConfig, ActJepaModel, ActionPredictorModel, Jepa
from robo_utils.train_utils import get_model_class


class TinyResNet(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_sizes=[hidden_size])
        self.proj = torch.nn.Conv2d(3, hidden_size, kernel_size=3, padding=1)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.proj(pixel_values))


def patch_resnet(monkeypatch):
    import models.act_model_original as act_model_original

    monkeypatch.setattr(
        act_model_original.ResNetModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: TinyResNet()),
    )


def tiny_config():
    horizon = {
        "observation.image": 1,
        "observation.state": 3,
        "episode_index": 1,
        "task_index": 1,
        "action": 3,
    }
    common = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "hidden_act": "relu",
        "attention_dropout": 0.0,
        "horizon": horizon,
    }
    return ActJepaConfig(
        action_dim=2,
        state_dim=5,
        encoder=common,
        target_encoder=common | {"causal": False},
        decoder=common | {"causal": False},
        predictor=common | {"causal": False},
        sigreg={"knots": 5, "num_proj": 16},
        loss_weights={
            "action": 1.0,
            "jepa": 1.0,
            "abstract": 1.0,
            "target_sigreg": 0.01,
            "context_sigreg": 0.01,
        },
        target_update="grad",
        horizon=horizon,
    )


def tiny_batch(batch_size=2):
    return {
        "observation.image": torch.randn(batch_size, 3, 8, 8),
        "observation.state": torch.randn(batch_size, 3, 5),
        "observation.state_is_pad": torch.zeros(batch_size, 3, dtype=torch.bool),
        "task_index": torch.zeros(batch_size, dtype=torch.long),
        "action": torch.randn(batch_size, 3, 2),
        "action_is_pad": torch.zeros(batch_size, 3, dtype=torch.bool),
        "labels": torch.randn(batch_size, 3, 2),
    }


def test_model_exports_resolve_action_predictor_config():
    config = Config.load("configs/pusht/action_predictor.yaml")

    assert models.Jepa is Jepa
    assert models.ActionPredictorModel is ActionPredictorModel
    assert get_model_class(config.model.class_name) is Jepa
    assert get_model_class(config.action_predictor.model.class_name) is ActionPredictorModel


def test_act_jepa_forward_backward_optimizer_step(monkeypatch):
    patch_resnet(monkeypatch)
    model = ActJepaModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    output = model(**tiny_batch())

    assert output.loss.ndim == 0
    assert output.abstract_loss.ndim == 0
    assert output.target_sigreg_loss.ndim == 0
    assert output.context_sigreg_loss.ndim == 0
    assert output.reconstruction_loss.ndim == 0

    output.loss.backward()
    optimizer.step()
