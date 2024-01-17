import os
import sys

sys.path.append(os.curdir)

from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split

import utils


class SegSubDataset(Dataset):
    def __init__(self, args):
        super(SegSubDataset, self).__init__()
        self.config = args['config']
        self.wandb = args['wandb']
        self.processor = args['processor']
        self.slices = args['slices']

        self.volume_min = -1215.0
        self.volume_max = 1930.0

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item = self.slices[idx]
        image = self.get_image(item)
        # label, instance_id_to_semantic_id = self.get_label(item)
        label = self.get_label(item)

        inputs = self.processor(
            images=image,
            segmentation_maps=label,
            # instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors='pt'
        )

        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return item, inputs

    def get_slice(self, item):
        volume = np.load(item['volume'], allow_pickle=True)

        if item['dim'] == '0':
            slice = volume[item['slice'], :, :]
        elif item['dim'] == '1':
            slice = volume[:, item['slice'], :]
        else:
            raise ValueError(f'Unknown dimension: {item["dim"]}')

        return torch.from_numpy(slice)

    def scale(self, image):
        image = (image - self.volume_min) / (self.volume_max - self.volume_min)

        return image

    def get_image(self, item):
        slice = self.get_slice(item)
        slice = self.scale(slice)
        image = torch.stack([slice for _ in range(self.wandb.config.num_channels)])
        # image = torch.unsqueeze(image, dim=0)

        return image

    def get_label(self, item):
        item['volume'] = item['volume'].replace('seismic', 'horizon_labels')
        label = self.get_slice(item)

        return label

    # def get_label(self, item):
    #     item['volume'] = item['volume'].replace('seismic', 'horizon_labels')
    #     slice = self.get_slice(item)
    #     label = slice - torch.min(slice)
    #     label = label.to(torch.uint8)
    #     instance_id_to_semantic_id = {int(i): 0 for i in torch.unique(label).tolist()}
    #
    #     return label, instance_id_to_semantic_id


# def collate_fn(batch):
#     pixel_values = torch.stack([el[1]['pixel_values'] for el in batch])
#     pixel_mask = torch.stack([el[1]['pixel_mask'] for el in batch])
#     class_labels = [el[1]['class_labels'] for el in batch]
#     mask_labels = [el[1]['mask_labels'] for el in batch]
#     slices = [el[0] for el in batch]
#
#     inputs = {
#         'pixel_values': pixel_values,
#         'pixel_mask': pixel_mask,
#         'class_labels': class_labels,
#         'mask_labels': mask_labels
#     }
#
#     return slices, inputs


def get_volumes(config, set):
    root_test = config['path']['data']['raw']['test']
    root_train = config['path']['data']['raw']['train']
    notebooks_test = os.path.join(os.pardir, root_test)  # get path from notebooks
    notebooks_train = os.path.join(os.pardir, root_train)  # get path from notebooks

    root = root_test if set == 'test' else root_train
    notebooks = notebooks_test if set == 'test' else notebooks_train
    path = root if os.path.exists(root) else notebooks

    search_volumes = os.path.join(path, '**', '*.npy')
    volumes = glob(search_volumes, recursive=True)
    volumes = [volume for volume in volumes if 'seismic_block' in volume or set == 'test']

    return volumes


def get_slices(wandb, volumes):
    slices = []

    for volume in volumes:
        slices += get_slices_from_volume(wandb, volume)

    if wandb.config.dataset_size:
        random.seed(wandb.config.random_state)
        slices = random.sample(slices, k=wandb.config.dataset_size)

    return slices


def get_slices_from_volume(wandb, volume):
    slices = []

    for dim in wandb.config.dim.split(','):
        if dim == '0' or dim == '1':
            slices += [{'volume': volume, 'dim': dim, 'slice': i} for i in range(300)]
        else:
            raise ValueError(f'Unknown dimension: {dim}')

    return slices


def get_training_slices(config, wandb):
    training_volumes = get_volumes(config, set='training')
    training_slices = get_slices(wandb, training_volumes)

    train_slices, val_slices = train_test_split(
        training_slices,
        test_size=wandb.config.val_size,
        random_state=wandb.config.random_state
    )

    return train_slices, val_slices


def get_submission_slices():
    return  # TODO


def compute_image_mean_std(config):
    def min_max_scaling(vol, min, max):
        vol = (vol - min) / (max - min)

        return vol

    volumes = get_volumes(config, set='train')

    mins, maxs = [], []
    print('\nCompute min and max :')
    for volume in tqdm(volumes):
        vol = np.load(volume, allow_pickle=True)
        mins.append(np.min(vol))
        maxs.append(np.max(vol))

    min = np.min(mins)
    max = np.max(maxs)

    means = []
    print('\nCompute mean:')
    for volume in tqdm(volumes):
        vol = np.load(volume, allow_pickle=True)
        vol = min_max_scaling(vol, min, max)
        means.append(np.mean(vol))

    mean = np.mean(means)

    vars = []
    print('\nCompute std:')
    for volume in tqdm(volumes):
        vol = np.load(volume, allow_pickle=True)
        vol = min_max_scaling(vol, min, max)
        vars.append(np.mean((vol - mean) ** 2))

    std = np.sqrt(np.mean(vars))

    print('\nmin', min)
    print('max', max)
    print('mean', mean)
    print('std', std)


def get_num_labels(config):
    labels = []
    volumes = get_volumes(config, 'training')

    for volume in volumes:
        label_path = volume.replace('seismic', 'horizon_labels')
        label = np.load(label_path, allow_pickle=True)
        labels += np.unique(label).tolist()
        labels = sorted(list(set(labels)))
        print(len(labels), labels)


if __name__ == '__main__':
    import src.models.make_lightning as ml
    from torch.utils.data import DataLoader

    config = utils.get_config()
    wandb = utils.init_wandb()

    processor, model = ml.get_processor_model(config, wandb)
    train_slices, val_slices = get_training_slices(config, wandb)

    args = {
        'config': config,
        'wandb': wandb,
        'processor': processor,
        'slices': train_slices[:100]
    }

    train_dataset = SegSubDataset(args)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False
    )

    min_value = 99
    max_value = 0
    for item, inputs in train_dataloader:
        min_value = min(min_value, inputs['labels'].min().item())
        max_value = max(max_value, inputs['labels'].max().item())
        print(min_value, max_value)
