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


def main(run_id, split):
    config = utils.get_config()
    run = utils.get_run_config(run_id)
    path_split = os.path.join(
        config['path']['data']['processed'][split],
        f"{run.name}-{run.id}",
    )
    os.makedirs(path_split, exist_ok=True)
    
    if split == 'train':
        multiprocess_make_mask(config, run, path_split)
    elif split == 'test':
        make_mask(
            config=config,
            device=torch.cuda.device(0),
            list_volume=md.get_volumes(config, set=split),
            run=run,
            path_split=path_split
        )


def multiprocess_make_mask(config, run, path_split):
    list_volume = md.get_volumes(config, set='train'),
    list_volume_split = split_list_volume(list_volume, torch.cuda.device_count())
    
    list_process = [
        Process(target=make_mask,
                kwargs={
                    'config': config,
                    'device': torch.cuda.device(i),
                    'list_volume': sub_list_volume,
                    'run': run,
                    'path_split': path_split
        })
        for i, sub_list_volume in enumerate(list_volume_split)
    ]
    for p in list_process:
        p.start()
    for p in list_process:
        p.join()


def split_list_volume(list_volume, nb_split):
    sub_len = len(list_volume) // nb_split
    list_volume_split = [list_volume[sub_len*i:sub_len*(i+1)] for i in range(nb_split-1)]
    list_volume_split.append(list_volume[sub_len*nb_split-1:])
    
    return list_volume_split


def make_mask(config, device, list_volume, run, path_split):
    segformer_model = load_segformer_model(config, run, device)

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
        volume = volume.to(device=device)
        
        return volume.to(dtype=torch.float16)
    
    def postprocess(binary_mask: torch.Tensor, shape):
        binary_mask = torch.moveaxis(binary_mask, 1, 2)
        binary_mask = tF.interpolate(binary_mask.unsqueeze(dim=1), size=shape[1:], mode="bilinear", align_corners=False)
        binary_mask = tF.sigmoid(binary_mask) > 0.5
        binary_mask = binary_mask.squeeze(dim=1).numpy(force=True)
        
        return binary_mask.astype(np.bool_)

    with torch.no_grad():
        for volume_path in tqdm(list_volume):
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
            

def load_segformer_model(config, run, device):
    train_volumes, val_volumes = md.get_training_volumes(config, run.config)
    processor = utils.get_processor(config, run.config)
    model = ml.get_model(run.config)

    args = {
        'config': config,
        'wandb_config': run.config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }
    
    path_checkpoint = os.path.join(config['path']['models']['root'], f'{run.name}-{run.id}.ckpt')
    lightning = ml.SegSubLightning.load_from_checkpoint(path_checkpoint, map_location=device, args=args)
    
    return lightning.to(dtype=torch.float16)
   
 
if __name__ == '__main__':
    main(run_id='k13mlcpr')