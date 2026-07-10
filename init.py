import os
from pathlib import Path

from dotenv import load_dotenv


def init():
    '''Load Hugging Face environment settings before training imports run.'''
    # WANDB_MODE set in the shell (or Docker ENV) must survive load_dotenv(override=True),
    # so one-off runs like `WANDB_MODE=offline python -m scripts.train ...` work.
    wandb_mode = os.getenv('WANDB_MODE')
    load_dotenv(override=True)
    if wandb_mode:
        os.environ['WANDB_MODE'] = wandb_mode

    hf_home = os.getenv('HF_HOME')
    if hf_home:
        path = Path(hf_home).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(path)