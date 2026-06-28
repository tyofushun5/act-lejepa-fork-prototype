# ACT-JEPA: Novel Joint-Embedding Predictive Architecture for Efficient Policy Representation Learning

[[Paper]](https://arxiv.org/abs/2501.14622) [[Code]](https://github.com/act-jepa/act-jepa) [[Project Website]](https://act-jepa.github.io/)

ACT-JEPA is an architecture designed to improve action prediction and world model understanding. The model learns to generate executable actions using IL, while simultaneously learning a latent world model using JEPA. The world model is developed by learning to predict future states in latent space, allowing the model to focus on high-level semantic information instead of irrelevant details. This approach enables efficient learning, develops a robust world model, and improves action prediction.

![architecture overview](assets/architecture-overview.svg)

## Quickstart

For a compact walkthrough of the ACT-JEPA architecture and tensor flow, start with [`act_jepa_illustrated.ipynb`](act_jepa_illustrated.ipynb).

The core ACT-JEPA/ACT-LEJEPA implementation is in [`models/act_jepa.py`](models/act_jepa.py#L233).
Baseline ACT-JEPA configs are in `configs/{environment}/act-jepa.yaml`, and
ACT-LEJEPA comparison configs are in `configs/{environment}/act-lejepa.yaml`.

## Repository Structure

```text
configs/           Training configs grouped by environment and model.
custom_envs/       Gymnasium wrappers and custom environment registration.
models/            ACT-JEPA, ACT, autoregressive transformer, and probes.
robo_utils/        Dataset, rollout, callback, and utility code.
scripts/           Training entry points.
transformer_utils/ Transformer layers shared by the models.
```
## Installation

```bash
conda create -n act-jepa python=3.12
conda activate act-jepa
pip install -r requirements.txt
```

## Training

Use `scripts.train` with any config in `configs/{environment}/{model}.yaml`:

```bash
python -m scripts.train --config_path configs/<environment>/<model>.yaml
```

For example, to train the baseline ACT-JEPA policy on Push-T use this:

```bash
python -m scripts.train --config_path configs/pusht/act-jepa.yaml
```

To train the ACT-LEJEPA comparison variant on Push-T use this:

```bash
python -m scripts.train --config_path configs/pusht/act-lejepa.yaml
```

`act-jepa.yaml` keeps the original ACT-JEPA target encoder behavior:
`model.target_update: ema`, no gradient through the target encoder, and
`EmaUpdateCallback`. `act-lejepa.yaml` uses `model.target_update: grad`, trains the
target encoder directly, and enables SIGReg on target latents. Set
`model.sigreg.weight` to tune or disable SIGReg.

Available environments are `pusht`, `metaworld`, and `mani_skill`. Available
model configs include `act`, `act-jepa`, `act-lejepa`, `ar_transformer`,
`state_predictor`, and `action_predictor`.

## Probe Training

Probe scripts are used to inspect what ACT-JEPA learns during training beyond the final rollout success rate.

### State Predictor

The state predictor freezes the learned encoder and trains a small head to reconstruct future state trajectories. This measures world-model understanding through RMSE and ATE.

```bash
python -m scripts.train_state_predictor \
  --config_path configs/pusht/state_predictor.yaml \
  --base_config_path configs/pusht/act-jepa.yaml
```

### Action Predictor

The action predictor reuses the JEPA-pretrained representation for action reconstruction. This tests whether latent dynamics learned from future-state prediction also transfer to control.

```bash
python -m scripts.train_action_predictor \
  --config_path configs/pusht/action_predictor.yaml
```

Equivalent probe configs are available under `configs/metaworld/` and `configs/mani_skill/`.

## Evaluation

Rollout evaluation is configured in the `env` section of each config and is usually run through `AgentEvaluatorCallback`.

You can also evaluate a trained checkpoint directly:

```bash
python -m scripts.evaluate --config_path configs/pusht/act-jepa.yaml
```

Pass `--checkpoint_path` to evaluate a specific checkpoint:

```bash
python -m scripts.evaluate \
  --config_path configs/pusht/act-jepa.yaml \
  --checkpoint_path path/to/model.safetensors
```

Important fields include:

- `rollout_steps`: evaluate every N training steps.
- `rollout_delay`: skip early rollouts.
- `num_episodes`: number of evaluation episodes.
- `env_names`: task names for multi-environment evaluation.
- `env_kwargs`: environment construction arguments.


## Datasets

Experiments use datasets hosted on Hugging Face:
[Push-T](https://huggingface.co/datasets/alek98/pusht),
[MetaWorld](https://huggingface.co/collections/alek98/metaworld), and
[ManiSkill](https://huggingface.co/collections/alek98/maniskill).
The datasets closely follow the LeRobot format for episode-indexed robotics data, expecting features such as:

- `observation.image`
- `observation.state`
- `action`
- `episode_index`
- `frame_index`
- `task_index`

### Custom Dataset

For custom data, use the same LeRobot-style features above, upload/cache it as a Hugging Face dataset, copy the closest config, and update:

```yaml
dataset:
  repo_ids: [your-username/dataset-name]
  revision: main       # or your dataset revision
  use_videos: true     # false if images are stored in parquet
  train_episodes_range: [0, 100]
  test_episodes_range: [100, 120]
```


## Citation

```bibtex
@article{vujinovic2025actjepa,
  title   = {ACT-JEPA: Novel Joint-Embedding Predictive Architecture for Efficient Policy Representation Learning},
  author  = {Vujinovic, Aleksandar and Kovacevic, Aleksandar},
  journal = {arXiv preprint arXiv:2501.14622},
  year    = {2025}
}
```
