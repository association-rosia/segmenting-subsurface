import os
import sys

sys.path.append(os.curdir)

from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

import utils


class SegSubDataset(Dataset):
    def __init__(self, args):
        super(SegSubDataset, self).__init__()
        self.config = args['config']
        self.wandb = args['wandb']
        self.processor = args['processor']
        self.volumes = args['volumes']
        self.slices = self.get_slices()

        self.volume_min = -1215.0
        self.volume_max = 1930.0

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item = self.slices[idx]
        image = self.get_image(item)
        label = self.get_label(item)

        inputs = self.processor(
            images=image,
            segmentation_maps=label,
            return_tensors='pt'
        )

        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        inputs['labels'] = self.get_binary_label(inputs['labels'])

        return item, inputs

    def get_slices(self):
        dims = self.wandb.config.dim.split(',')
        dilation = self.wandb.config.dilation
        slices = []

        for volume in self.volumes:
            for dim in dims:
                if dim == '0' or dim == '1':
                    slices += [{'volume': volume, 'dim': dim, 'slice': i} for i in range(0, 300, dilation)]
                else:
                    raise ValueError(f'Unknown dimension: {dim}')

        return slices

    def get_slice(self, item, dtype):
        volume = np.load(item['volume'], allow_pickle=True)

        if item['dim'] == '0':
            slice = volume[item['slice'], :, :]
        elif item['dim'] == '1':
            slice = volume[:, item['slice'], :]
        else:
            raise ValueError(f'Unknown dimension: {item["dim"]}')

        slice = torch.from_numpy(slice)
        slice = slice.to(dtype=dtype)
        slice = slice.T

        return slice

    def scale(self, image):
        image = (image - self.volume_min) / (self.volume_max - self.volume_min)

        return image

    def get_image(self, item):
        slice = self.get_slice(item, dtype=torch.float32)
        slice = self.scale(slice)
        image = torch.stack([slice for _ in range(self.wandb.config.num_channels)])

        return image

    def get_label(self, item):
        item['volume'] = item['volume'].replace('seismic', 'horizon_labels')
        label = self.get_slice(item, dtype=torch.uint8)

        return label

    def get_binary_label(self, label, kernel=3):
        label = label.float()
        padding_size = kernel
        padded_label = F.pad(label, (padding_size, padding_size, padding_size, padding_size), value=1)
        unfolded = padded_label.unfold(0, kernel, 1).unfold(1, kernel, 1)
        binary_label = (unfolded.std(dim=(2, 3)) == 0).byte()
        binary_label = 1 - binary_label[1:label.shape[0] + 1, 1:label.shape[1] + 1]

        self.plot_slice(binary_label)

        return binary_label

    def plot_slice(self, slice):
        ax = plt.subplot()

        if slice.shape[0] == 3:
            im = ax.imshow(slice[0], cmap='gray')
        else:
            im = ax.imshow(slice, interpolation='nearest')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()


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


def get_training_volumes(config, wandb):
    training_volumes = get_volumes(config, set='training')
    train_volumes, val_volumes = train_test_split(training_volumes, test_size=wandb.config.val_size)

    return train_volumes, val_volumes


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
        vol = vol.astype(np.float64)
        vol = min_max_scaling(vol, min, max)
        vars.append(np.mean((vol - mean) ** 2))

    std = np.sqrt(np.mean(vars))

    print('\nmin', min)
    print('max', max)
    print('mean', mean)
    print('std', std)


if __name__ == '__main__':
    import src.models.make_lightning as ml
    from torch.utils.data import DataLoader

    config = utils.get_config()
    # compute_image_mean_std(config)
    wandb = utils.init_wandb()

    processor, model = ml.get_processor_model(config, wandb)
    train_volumes = get_volumes(config, set='train')

    args = {
        'config': config,
        'wandb': wandb,
        'processor': processor,
        'volumes': train_volumes,
    }

    train_dataset = SegSubDataset(args)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False
    )

    for item, inputs in train_dataloader:
        break
