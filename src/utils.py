import os
import torch
import wandb
import yaml
from transformers import AutoImageProcessor


def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def get_config() -> dict:
    root = os.path.join('config', 'config.yml')
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def init_wandb(yml_file):
    root = os.path.join('config', yml_file)
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    project_config = get_config()
    
    wandb.init(
        entity=project_config['wandb']['entity'],
        project=project_config['wandb']['project'],
        config=config
    )

    return wandb.config


def get_processor(config, wandb_config):
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb_config.model_id,
        do_rescale=False,
        image_mean=config['data']['mean'],
        image_std=config['data']['std'],
        do_reduce_labels=False,
        do_pad=False
    )

    return processor


def get_run_config(run_id: str):
    project_config = get_config()
    
    api = wandb.Api()
    run = wandb.apis.public.Run(
        client=api.client,
        entity=project_config['wandb']['entity'],
        project=project_config['wandb']['project'],
        run_id=run_id,
    )

    return run
