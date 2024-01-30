import os
import torch
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as tvF
import torch.nn.functional as tF

import src.models.segformer.make_lightning as segformer_ml
import src.data.make_dataset as md

from src import utils

def make_mask(config, run_id):
    run = utils.get_run_config(run_id)
    
    segformer_model = load_segformer_model(config, run)

    volume_min = config['data']['min']
    volume_max = config['data']['max']
    contrast_factor = run.config['contrast_factor']
    processor = utils.get_processor(config, run.config)
    
    def preprocess(volume: np.ndarray):
        volume = (volume - volume_min) / (volume_max - volume_min)
        volume = np.moveaxis(volume, 1, 2)
        volume = torch.from_numpy(volume).unsqueeze(1)
        volume = tvF.adjust_contrast(volume, contrast_factor=contrast_factor)
        volume = torch.repeat_interleave(volume, repeats=3, dim=1)
        volume = processor(images=volume, return_tensors='pt')
        volume = volume.to(device='cuda')
        
        return volume.to(dtype=torch.float16)
    
    def postprocess(binary_mask: torch.Tensor, shape):
        binary_mask = torch.moveaxis(binary_mask, 1, 2)
        binary_mask = tF.interpolate(binary_mask.unsqueeze(dim=1), size=shape[1:], mode="bilinear", align_corners=False)
        binary_mask = binary_mask.squeeze(dim=1).numpy(force=True)
        
        return binary_mask

    with torch.no_grad():
        for split in ['train', 'test']:
            path_split = os.path.join(
                config['path']['data']['processed'][split],
                f"{run.name}-{run.id}",
            )
            os.makedirs(path_split, exist_ok=True)
            for volume_path in tqdm(md.get_volumes(config, set=split)):
                volume_name = os.path.basename(volume_path)
                binary_mask_path = os.path.join(path_split, volume_name)
                binary_mask_path = binary_mask_path.replace('seismic', 'binary_mask')
                if os.path.exists(binary_mask_path):
                    continue
                volume = np.load(volume_path, allow_pickle=True)
                shape = volume.shape
                volume = preprocess(volume)
                binary_mask = segformer_model(volume)
                binary_mask = postprocess(binary_mask, shape)
                np.save(binary_mask_path, binary_mask, allow_pickle=True)
            

def load_segformer_model(config, run):
    train_volumes, val_volumes = md.get_training_volumes(config, run.config)
    processor = utils.get_processor(config, run.config)
    model = segformer_ml.get_model(run.config)

    args = {
        'config': config,
        'wandb_config': run.config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }
    
    path_checkpoint = os.path.join(config['path']['models']['root'], f'{run.name}-{run.id}.ckpt')
    lightning = segformer_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, args=args)
    
    return lightning.to(dtype=torch.float16)
   
 
if __name__ == '__main__':
    config = utils.get_config()
    run_id = 'k13mlcpr'
    make_mask(config, run_id)