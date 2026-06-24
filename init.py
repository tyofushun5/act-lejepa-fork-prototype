import os
from pathlib import Path

from dotenv import load_dotenv


def init():
    '''Load Hugging Face environment settings before training imports run.'''
    load_dotenv(override=True)

    hf_home = os.getenv('HF_HOME')
    if hf_home:
        path = Path(hf_home).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(path)