import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as tvF
import wandb
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoImageProcessor

import src.models.mask2former.make_lightning as mask2former_ml
import src.models.segformer.make_lightning as segformer_ml
import src.models.segment_anything.make_lightning as segment_anything_ml


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
        config = yaml.safe_load(f)

    wandb.init(
        entity='association-rosia',
        project='segmenting-subsurface',
        config=config
    )

    return wandb.config


def resize_tensor_2d(tensor, size):
    resized_tensor = tvF.resize(tensor.unsqueeze(0), size=size).squeeze(0)

    return resized_tensor


def plot_slice(slice):
    ax = plt.subplot()

    if slice.shape[0] == 3:
        im = ax.imshow(slice[0], cmap='gray')
    else:
        im = ax.imshow(slice, interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

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
    project_config = get_config()

    api = wandb.Api()
    run = wandb.apis.public.Run(
        client=api.client,
        entity=project_config['wandb']['entity'],
        project=project_config['wandb']['project'],
        run_id=run_id,
    )

    return run


def load_segformer(config, run):
    processor = get_processor(config, run.config)
    model = segformer_ml.get_model(run.config)

    args = {
        'config': config,
        'wandb_config': run.config,
        'model': model,
        'processor': processor,
        'train_volumes': None,
        'val_volumes': None
    }

    path_checkpoint = os.path.join(config['path']['models']['root'], f'{run.name}-{run.id}.ckpt')
    lightning = segformer_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, args=args)

    device = get_device()
    lightning = lightning.to(torch.float16)
    lightning = lightning.to(device)

    return lightning


def load_mask2former(config, run):
    processor = get_processor(config, run.config)
    model = mask2former_ml.get_model(run.config)

    args = {
        'config': config,
        'wandb_config': run.config,
        'model': model,
        'processor': processor,
        'train_volumes': None,
        'val_volumes': None
    }

    path_checkpoint = os.path.join(config['path']['models']['root'], f'{run.name}-{run.id}.ckpt')
    lightning = mask2former_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, args=args)

    device = get_device()
    lightning = lightning.to(torch.float16)
    lightning = lightning.to(device)

    return lightning


def load_segment_anything(config, run):
    processor = get_processor(config, run.config)
    model = segment_anything_ml.get_model(run.config)

    args = {
        'config': config,
        'wandb_config': run.config,
        'model': model,
        'processor': processor,
        'train_volumes': None,
        'val_volumes': None
    }

    path_checkpoint = os.path.join(config['path']['models']['root'], f'{run.name}-{run.id}.ckpt')
    lightning = segment_anything_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, args=args)

    device = get_device()
    lightning = lightning.to(torch.float16)
    lightning = lightning.to(device)

    return lightning
