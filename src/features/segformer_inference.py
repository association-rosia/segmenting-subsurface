import os
import torch
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as tvF
import torch.nn.functional as tF

import src.models.segformer.make_lightning as ml
import src.data.make_dataset as md
from multiprocessing import Process

from src import utils


def main(run_id):
    config = utils.get_config()
    run = utils.get_run_config(run_id)
    for split in ['train', 'test']:
        path_split = os.path.join(
            config['path']['data']['processed'][split],
            f"{run.name}-{run.id}",
        )
        os.makedirs(path_split, exist_ok=True)
        multiprocess_make_mask(config, run, split)


def multiprocess_make_mask(config, run, split):
    list_volume = md.get_volumes(config, set=split)
    list_volume_split = split_list_volume(list_volume, torch.cuda.device_count())
    
    list_process = [
        Process(target=SegformerInference(
                    config=config,
                    cuda_idx=i,
                    list_volume=sub_list_volume,
                    run=run,
                    split=split
        ))
        for i, sub_list_volume in enumerate(list_volume_split)
    ]
    for p in list_process:
        p.start()
    for p in list_process:
        p.join()


def split_list_volume(list_volume, nb_split):
    sub_len = len(list_volume) // nb_split
    list_volume_split = [list_volume[sub_len*i:sub_len*(i+1)] for i in range(nb_split-1)]
    list_volume_split.append(list_volume[sub_len*(nb_split-1):])
    
    return list_volume_split


class SegformerInference:
    def __init__(self, config, cuda_idx, list_volume, run, split) -> None:
        self.config = config
        self.device = f'cuda:{cuda_idx}'
        self.list_volume = list_volume
        self.run = run
        self.split = split
        self.volume_min = config['data']['min']
        self.volume_max = config['data']['max']
        self.contrast_factor = run.config['contrast_factor']
        self.processor = utils.get_processor(config, run.config)
 
    def __call__(self):
        model = self.load_model()

        with torch.no_grad():
            for volume_path in tqdm(self.list_volume):
                volume_name = os.path.basename(volume_path)
                binary_mask_path = self.get_mask_path(volume_name)
                if os.path.exists(binary_mask_path):
                    continue
                
                volume = np.load(volume_path, allow_pickle=True)
                shape = volume.shape
                volume = self.preprocess(volume)
                binary_mask = model(volume)
                binary_mask = self.postprocess(binary_mask, shape)
                np.save(binary_mask_path, binary_mask, allow_pickle=True)

    def get_mask_path(self, volume_name):
        path = os.path.join(
            self.config['path']['data']['processed'][self.split],
            f"{self.run.name}-{self.run.id}",
        )
        path = os.path.join(path, volume_name)
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
        volume = torch.repeat_interleave(volume, repeats=3, dim=1)
        volume = self.processor(images=volume, return_tensors='pt')
        volume = volume.to(device=self.device, dtype=torch.float16)
        
        return volume

    @staticmethod
    def postprocess(binary_mask: torch.Tensor, shape):
        binary_mask = torch.moveaxis(binary_mask, 1, 2)
        binary_mask = tF.interpolate(binary_mask.unsqueeze(dim=1), size=shape[1:], mode="bilinear", align_corners=False)
        binary_mask = tF.sigmoid(binary_mask) > 0.5
        binary_mask = binary_mask.squeeze(dim=1).numpy(force=True)
        
        return binary_mask
            

    def load_model(self):
        train_volumes, val_volumes = md.get_training_volumes(self.config, self.run.config)
        processor = utils.get_processor(self.config, self.run.config)
        model = ml.get_model(self.run.config)

        args = {
            'config': self.config,
            'wandb_config': self.run.config,
            'model': model,
            'processor': processor,
            'train_volumes': train_volumes,
            'val_volumes': val_volumes
        }
        
        path_checkpoint = os.path.join(self.config['path']['models']['root'], f'{self.run.name}-{self.run.id}.ckpt')
        lightning = ml.SegSubLightning.load_from_checkpoint(path_checkpoint, map_location=self.device, args=args)
        
        return lightning.to(dtype=torch.float16)
   
 
if __name__ == '__main__':
    main(run_id='k13mlcpr')