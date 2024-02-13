import os
import warnings
from multiprocessing import Process

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from tqdm import tqdm

import src.data.make_dataset as md
import src.models.mask2former.make_lightning as ml
from src import utils

warnings.filterwarnings('ignore')


def main(run_id):
    config = utils.get_config()
    run = utils.get_run(run_id)
    path_split = os.path.join(config['path']['data']['processed']['test'], f'{run.name}-{run.id}', )
    os.makedirs(path_split, exist_ok=True)
    multiprocess_make_mask(config, run, split='test')


def multiprocess_make_mask(config, run, split, batch=300):
    list_volume = md.get_volumes(config, set=split)
    list_volume_split = split_list_volume(list_volume, torch.cuda.device_count())

    list_process = [
        Process(target=Mask2formerInference(
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


class Mask2formerInference:
    def __init__(self, config, cuda_idx, list_volume, run, split, batch=300) -> None:
        self.config = config
        self.device = f'cuda:{cuda_idx}'
        self.list_volume = list_volume
        self.run = run
        self.split = split
        self.batch = batch
        self.volume_min = config['data']['min']
        self.volume_max = config['data']['max']
        self.contrast_factor = run.config['contrast_factor']
        self.processor = utils.get_processor(config, run.config)

    def __call__(self):
        with torch.no_grad():
            model = self.load_model()
            for volume_path in tqdm(self.list_volume):
                volume_name = os.path.basename(volume_path)
                instance_mask_path = self.get_mask_path(volume_name)

                if os.path.exists(instance_mask_path):
                    continue

                volume = np.load(volume_path, allow_pickle=True)
                binary_mask = self.load_binary_mask(volume_name)
                instance_mask = self.predict(volume, binary_mask, model)
                np.save(instance_mask_path, instance_mask, allow_pickle=True)
        
            del model.model, model
        torch.cuda.empty_cache()

    def get_folder_path(self):
        path = os.path.join(
            self.config['path']['data']['processed'][self.split],
            f'{self.run.name}-{self.run.id}'
        )
        
        return path
    
    def get_mask_path(self, volume_name):
        folder_path = self.get_folder_path()

        if self.split == 'train':
            volume_name = volume_name.replace('seismic', 'instance_mask')
        else:
            volume_name = volume_name.replace('test', 'sub')

        path = os.path.join(folder_path, volume_name)

        return path

    def preprocess(self, volume: torch.Tensor, binary_mask: torch.Tensor):
        volume = (volume - self.volume_min) / (self.volume_max - self.volume_min)
        volume = torch.moveaxis(volume, 1, 2)
        volume = volume.unsqueeze(1)
        volume = tvF.adjust_contrast(volume, contrast_factor=self.contrast_factor)
        volume = volume.squeeze()
        binary_mask = torch.moveaxis(binary_mask, 1, 2)

        images = self.build_images(volume, binary_mask)
        images = torch.unbind(images, dim=0)
        inputs = self.processor(images=images, return_tensors='pt')
        inputs = inputs.to(device=self.device, dtype=torch.float16)

        return inputs
    
    def predict(self, volume: torch.Tensor, binary_mask: torch.Tensor, model: torch.nn.Module):
        list_instance_mask = []
        volume = torch.from_numpy(volume)
        binary_mask = torch.from_numpy(binary_mask)
        sub_inputs = zip(
            torch.chunk(volume, volume.shape[0] // self.batch),
            torch.chunk(binary_mask, binary_mask.shape[0] // self.batch)
        )
        for sub_volume, sub_binary_mask in sub_inputs:
            inputs = self.preprocess(sub_volume, sub_binary_mask)
            outputs = model(inputs)
            sub_instance_mask = self.postprocess(outputs, sub_volume.shape)
            list_instance_mask.append(sub_instance_mask)
        
        instance_mask = np.concatenate(list_instance_mask)

        return instance_mask

    def postprocess_output(self, output, size):
        values = torch.unique(output).tolist()

        if 255 in values:
            output[output == 255] = values[-2] + 1

        if -1 in values:
            output[output == -1] = torch.max(output) + 1

        output = torch.moveaxis(output, 0, 1)
        output = utils.resize_tensor_2d(output, size=size, interpolation=tvF.InterpolationMode.NEAREST_EXACT)
        output = output.to(torch.uint8)

        return output

    def postprocess(self, outputs, shape):
        outputs = self.processor.post_process_instance_segmentation(outputs)
        instance_mask = torch.stack([self.postprocess_output(output['segmentation'], shape[1:]) for output in outputs])
        instance_mask = instance_mask.numpy(force=True)

        return instance_mask

    def load_binary_mask(self, volume_name):
        path = os.path.join(
            self.config['path']['data']['processed'][self.split],
            self.run.config['model_mask_id'],
        )
        binary_mask_path = os.path.join(path, volume_name)
        if self.split == 'train':
            binary_mask_path = binary_mask_path.replace('seismic', 'binary_mask')
        else:
            binary_mask_path = binary_mask_path.replace('vol', 'bmask')
        binary_mask = np.load(binary_mask_path, allow_pickle=True)

        return binary_mask

    def build_images(self, volume, binary_mask):
        num_channels_mask = self.run.config['num_channels_mask']

        images = None
        if num_channels_mask == 0:
            images = torch.repeat_interleave(volume, repeats=3, dim=1)
        elif num_channels_mask == 1:
            images = torch.stack([volume, binary_mask, volume], dim=1)
        elif num_channels_mask == 2:
            images = torch.stack([binary_mask, volume, binary_mask], dim=1)
        elif num_channels_mask == 3:
            images = torch.repeat_interleave(binary_mask, repeats=3, dim=1)
        else:
            ValueError(f'Wrong num_channels_mask value: {num_channels_mask}')

        return images

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
    main(run_id='xzs93mfw')
