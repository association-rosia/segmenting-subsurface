import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as tvF
import wandb
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoImageProcessor


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
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


def init_wandb(yml_file):
    root = os.path.join('config', yml_file)
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        wandb_config = yaml.safe_load(f)

    config = get_config()

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        config=wandb_config
    )

    return wandb.config


def resize_tensor_2d(tensor, size, interpolation=tvF.InterpolationMode.BILINEAR):
    resized_tensor = tvF.resize(tensor.unsqueeze(0), size=size, interpolation=interpolation).squeeze(0)

    return resized_tensor


def plot_slice(slice, title=None):
    ax = plt.subplot()

    if slice.shape[0] == 3 or slice.shape[0] == 1:
        im = ax.imshow(slice[0], cmap='gray')
    else:
        im = ax.imshow(slice, interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(title)
    plt.show()


def get_processor(config, wandb_config):
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb_config['model_id'],
        do_rescale=False,
        image_mean=config['data']['mean'],
        image_std=config['data']['std'],
        do_reduce_labels=False,
        reduce_labels=False,
        do_pad=False
    )

    return processor


def get_run(run_id: str):
    run = None

    if run_id:
        project_config = get_config()

        api = wandb.Api()
        run = wandb.apis.public.Run(
            client=api.client,
            entity=project_config['wandb']['entity'],
            project=project_config['wandb']['project'],
            run_id=run_id,
        )

    return run


class RunDemo:
    def __init__(self, config_file, id, name) -> None:
        self.config = self.get_config(config_file)
        self.name = name
        self.id = id
    
    @staticmethod
    def get_config(config_file) -> dict:
        root = os.path.join('config', config_file)
        notebooks = os.path.join(os.pardir, root)
        path = root if os.path.exists(root) else notebooks

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config
