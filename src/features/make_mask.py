import os
import torch
import numpy as np

import torchvision.transforms.functional as tvF

import src.models.segformer.make_lightning as segformer_ml
import src.models.segment_anything.make_lightning as sam_ml
import src.data.make_dataset as segformer_md

from src import utils


def make_mask(config, run_id):
    run = utils.get_run_config(run_id)
    
    segformer_model = load_sam_model(config, run)

    volume_min = config['data']['min']
    volume_max = config['data']['max']
    contrast_factor = run['config']['contrast_factor']
    processor = sam_ml.get_processor(run['config'])
    
    def preprocess(volume):
        volume = (volume - volume_min) / (volume_max - volume_min)
        volume = tvF.adjust_contrast(volume, contrast_factor=contrast_factor)
        volume = processor(images=volume, return_tensors='pt')
        
        return volume

    for split in ['train', 'test']:
        path_split = os.path.join(
            config['path']['data']['processed'][split],
            f"{run['name']}-{run['id']}",
        )
        os.makedirs(path_split, exist_ok=True)
        for volume_path in segformer_md.get_volumes(config, set=split):
            volume = np.load(volume_path, allow_pickle=True)
            volume = preprocess(volume)
            outputs = segformer_model(volume)
            prompt_volume = outputs.argmax(dim=1).numpy(force=True)
            volume_name = os.path.basename(volume_path)
            prompt_volume_path = os.path.join(path_split, volume_name)
            np.save(prompt_volume_path, prompt_volume, allow_pickle=True)
            

def load_sam_model(config, run):
    train_volumes, val_volumes = sam_ml.get_training_volumes(config, val_size=run['config'])
    processor = sam_ml.get_processor(config, run['config'])
    model = sam_ml.get_model(run['config'])

    args = {
        'config': config,
        'wandb_config': run['config'],
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }
    
    path_checkpoint = os.path.join(config['path']['models'], f"{run['name']}-{run['id']}.ckpt")
    
    device = torch.device(utils.get_device())
    lightning = sam_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, map_location=device, args=args)

    return lightning


def load_mask2former_model(config, run):
    train_volumes, val_volumes = segformer_md.get_training_volumes(config, val_size=run['config'])
    processor = sam_ml.get_processor(config, run['config'])
    model = sam_ml.get_model(run['config'])

    args = {
        'config': config,
        'wandb': run['config'],
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }
    
    path_checkpoint = os.path.join(config['path']['models'], f"{run['name']}-{run['id']}.ply")
    
    device = torch.device(utils.get_device())
    lightning = segformer_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, map_location=device, args=args)

    return lightning
   
 
if __name__ == '__main__':
    config = utils.get_config()
    run_id = 'k13mlcpr'
    make_mask(config, run_id)