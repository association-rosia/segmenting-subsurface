import os

import torch
import wandb
import yaml


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def get_config():
    root = os.path.join('config', 'config.yml')
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def init_wandb():
    root = os.path.join('config', 'wandb.yml')
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(
        entity='association-rosia',
        project='segmenting-subsurface',
        config=config
    )

    return wandb
