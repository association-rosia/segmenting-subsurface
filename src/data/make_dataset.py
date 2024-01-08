import os
import sys

sys.path.append(os.curdir)

from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Mask2FormerImageProcessor

import torchvision.transforms.functional as F

import utils


class SegSubDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.items = self.get_items()

        self.processor = Mask2FormerImageProcessor(
            num_labels=1,
            do_resize=False,
            do_rescale=False,
            image_mean=args['config']['data']['stats']['mean'],
            image_std=args['config']['data']['stats']['std']
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        image = self.get_image(item)
        image = self.scale(image)
        image = self.resize(image)
        inputs = self.processor.preprocess(image)

        label = self.get_label(item)
        label = self.resize(label)

        return inputs, label

    def get_slice(self, item):
        volume = np.load(item['volume'], allow_pickle=True)

        if item['dim'] == '0':
            slice = volume[item['slice'], :, :]
        elif item['dim'] == '1':
            slice = volume[:, item['slice'], :]
        else:
            raise ValueError(f'Unknown dimension: {item["dim"]}')

        return torch.from_numpy(slice)

    def get_image(self, item):
        slice = self.get_slice(item)
        image = torch.stack([slice, slice, slice], dim=0)

        return image

    def scale(self, image):
        min = self.args['config']['data']['stats']['min']
        max = self.args['config']['data']['stats']['max']
        image = (image - min) / (max - min)

        return image

    def resize(self, image):
        image = F.resize(image, (self.args['config']['data']['size'], self.args['config']['data']['size']))

        return image

    def get_label(self, item):
        item['volume'] = item['volume'].replace('seismic', 'horizon_labels')
        slice = self.get_slice(item)
        label = slice - torch.min(slice)
        label = torch.stack([torch.zeros(label.shape), label], dim=0)

        return label

    def get_items(self):
        slices = []
        volumes = self.args['volumes']

        for volume in volumes:
            # vol = np.load(volume, allow_pickle=True)
            slices += self.get_items_from_volume(volume)

        return slices

    def get_items_from_volume(self, volume):
        items = []

        for dim in self.args['dim'].split(','):
            if dim == '0' or dim == '1':
                # volume shape: (300, 300, 100)
                items += [{'volume': volume, 'dim': dim, 'slice': i} for i in range(300)]
            else:
                raise ValueError(f'Unknown dimension: {dim}')

        return items


def get_volumes(config, set):
    path = config['path']['data']['raw']['test'] if set == 'test' else config['path']['data']['raw']['train']
    search_volumes = os.path.join(path, '**', '*.npy')
    volumes = glob(search_volumes, recursive=True)
    volumes = [volume for volume in volumes if 'seismic_block' in volume or set == 'test']

    return volumes


def compute_image_mean_std(config):
    def min_max_scaling(vol, min, max):
        vol = (vol - min) / (max - min)

        return vol

    volumes = get_volumes(config, set='train')

    mins = []
    print('\nCompute min:')
    for volume in tqdm(volumes):
        vol = np.load(volume, allow_pickle=True)
        mins.append(np.min(vol))

    min = np.min(mins)

    maxs = []
    print('\nCompute max:')
    for volume in tqdm(volumes):
        vol = np.load(volume, allow_pickle=True)
        maxs.append(np.max(vol))

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


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    config = utils.get_config()

    # compute_image_mean_std(config)

    set = 'train'
    train_volumes = get_volumes(config, set=set)
    args = {'config': config, 'set': set, 'volumes': train_volumes, 'dim': '0,1'}
    train_dataset = SegSubDataset(args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)

    for inputs, label in train_dataloader:
        print('\npixel_values', inputs['pixel_values'][0].shape)
        print('pixel_mask', inputs['pixel_mask'][0].shape)
        print('label', label.shape)
        break
