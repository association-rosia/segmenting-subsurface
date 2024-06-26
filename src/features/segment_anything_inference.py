import os
from multiprocessing import Process

import numpy as np
import torch
import torch.nn.functional as tF
import torchvision.transforms.functional as tvF
from tqdm import tqdm

import src.data.make_dataset as md
import src.models.segment_anything.make_lightning as ml
from src import utils
import yaml


def main(run_id):
    config = utils.get_config()
    run = utils.get_run(run_id)

    for split in ['train', 'test']:
        path_split = get_path_split(config, split, run)
        os.makedirs(path_split, exist_ok=True)
        multiprocess_make_mask(config, run, split)


def get_wandb_config(config_file) -> dict:
    root = os.path.join('config', config_file)
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_path_split(config, split, run):
    if run:
        path_split = os.path.join(config['path']['data']['processed'][split], f'{run.name}-{run.id}')
    else:
        wandb_config = get_wandb_config('segment_anything.yml')
        folder_path = wandb_config['model_id'].replace('/', '_')
        path_split = os.path.join(config['path']['data']['processed'][split], folder_path)

    return path_split


def multiprocess_make_mask(config, run, split, batch=15):
    list_volume = md.get_volumes(config, set=split)
    list_volume_split = split_list_volume(list_volume, torch.cuda.device_count())

    list_process = [
        Process(target=SAMInference(
            config=config,
            cuda_idx=i,
            list_volume=sub_list_volume,
            run=run,
            split=split,
            batch=batch
        ))
        for i, sub_list_volume in enumerate(list_volume_split)
    ]

    for p in list_process:
        p.start()

    for p in list_process:
        p.join()


def split_list_volume(list_volume, nb_split):
    sub_len = len(list_volume) // nb_split
    list_volume_split = [list_volume[sub_len * i:sub_len * (i + 1)] for i in range(nb_split - 1)]
    list_volume_split.append(list_volume[sub_len * (nb_split - 1):])

    return list_volume_split


class SAMInference:
    def __init__(self, config, cuda_idx, list_volume, run, split, batch=5) -> None:
        self.config = config
        self.device = f'cuda:{cuda_idx}'
        self.list_volume = list_volume
        self.run = run
        self.split = split
        self.volume_min = config['data']['min']
        self.volume_max = config['data']['max']
        self.wandb_config = self.get_wandb_config()
        self.contrast_factor = self.wandb_config['contrast_factor']
        self.batch = batch

    def __call__(self):
        with torch.no_grad():
            model = self.load_model()

            for volume_path in tqdm(self.list_volume):
                volume_name = os.path.basename(volume_path)
                binary_mask_path = self.get_mask_path(volume_name)

                if os.path.exists(binary_mask_path):
                    continue

                volume = np.load(volume_path, allow_pickle=True)
                volume_shape = volume.shape
                volume = self.preprocess(volume)
                binary_mask = self.predict(volume, model)
                binary_mask = self.postprocess(binary_mask, volume_shape)
                np.save(binary_mask_path, binary_mask, allow_pickle=True)
            
            del model
        torch.cuda.empty_cache()

    def get_folder_path(self):
        if self.run:
            folder_name = f'{self.run.name}-{self.run.id}'
        else:
            folder_name = self.wandb_config['model_id'].replace('/', '_')
            
        path = os.path.join(
            self.config['path']['data']['processed'][self.split],
            folder_name
        )

        return path

    def get_wandb_config(self):
        if self.run:
            wandb_config = self.wandb_config
        else:
            wandb_config = get_wandb_config('segment_anything.yml')

        return wandb_config

    def get_mask_path(self, volume_name):
        folder_path = self.get_folder_path()
        path = os.path.join(folder_path, volume_name)

        if self.split == 'train':
            path = path.replace('seismic', 'binary_mask')
        else:
            path = path.replace('vol', 'bmask')

        return path

    def preprocess(self, volume: np.ndarray):
        volume = (volume - self.volume_min) / (self.volume_max - self.volume_min)
        volume = np.moveaxis(volume, 1, 2)
        volume = torch.from_numpy(volume).unsqueeze(1)
        volume = tvF.adjust_contrast(volume, contrast_factor=self.contrast_factor)
        volume = tvF.resize(volume, (1024, 1024))
        volume = torch.repeat_interleave(volume, repeats=3, dim=1)

        return volume

    def predict(self, volume: torch.Tensor, model: torch.nn.Module):
        list_binary_mask = []
        for sub_volume in torch.chunk(volume, volume.shape[0] // self.batch):
            sub_volume = sub_volume.to(device=self.device, dtype=torch.float16)
            outputs = model(pixel_values=sub_volume, multimask_output=False)
            sub_binary_mask = torch.squeeze(outputs['pred_masks'].cpu())
            list_binary_mask.append(sub_binary_mask)

        binary_mask = torch.concatenate(list_binary_mask)

        return binary_mask

    @staticmethod
    def postprocess(binary_mask: torch.Tensor, shape):
        binary_mask = torch.moveaxis(binary_mask, 1, 2)
        binary_mask = tvF.resize(binary_mask, size=shape[-2:])
        binary_mask = tF.sigmoid(binary_mask) > 0.5
        binary_mask = binary_mask.numpy(force=True)

        return binary_mask

    def load_model(self):
        train_volumes, val_volumes = md.get_training_volumes(self.config, self.wandb_config)
        processor = utils.get_processor(self.config, self.wandb_config)
        model = ml.get_model(self.wandb_config)

        args = {
            'config': self.config,
            'wandb_config': self.wandb_config,
            'model': model,
            'processor': processor,
            'train_volumes': train_volumes,
            'val_volumes': val_volumes
        }

        if self.run:
            path_checkpoint = os.path.join(self.config['path']['models']['root'], f'{self.run.name}-{self.run.id}.ckpt')
            lightning = ml.SegSubLightning.load_from_checkpoint(path_checkpoint, map_location=self.device, args=args)
        else:
            lightning = ml.SegSubLightning(args)

        model = lightning.model.to(device=self.device, dtype=torch.float16)

        return model


if __name__ == '__main__':
    main(run_id=None)
