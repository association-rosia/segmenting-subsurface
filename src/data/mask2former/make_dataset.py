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
import torch.nn.functional as tF
import torchvision.transforms.functional as tvF
from src import utils


class SegSubDataset(Dataset):
    def __init__(self, args):
        super(SegSubDataset, self).__init__()
        self.config = args['config']
        self.wandb_config = args['wandb_config']
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
        inputs['labels'] = self.process_label(inputs['labels'])

        return item, inputs

    def get_slices(self):
        dims = self.wandb_config['dim'].split(',')
        dilation = self.wandb_config['dilation']
        slices = []

        for volume in self.volumes:
            for dim in dims:
                if dim == '0' or dim == '1':
                    slices += [{'volume': volume, 'dim': dim, 'slice': i} for i in range(0, 300, dilation)]
                else:
                    raise ValueError(f'Unknown dimension: {dim}')

        return slices
    
    def get_binary_slice(self, item, dtype):
        volume_name = os.path.basename(item['volume'])
        volume_path = os.path.join(self.config['path']['data']['processed']['train'], self.wandb_config['model_prompt'], volume_name)
        volume = np.load(volume_path, allow_pickle=True)

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
        binary_slice = self.get_binary_slice(item, dtype=torch.float32)
        slice = self.scale(slice)
        image = torch.stack([slice for _ in range(self.wandb_config['num_channels'])])
        contrast_factor = self.wandb_config['contrast_factor']
        image = tvF.adjust_contrast(image, contrast_factor=contrast_factor)
        image[3, :, :] = binary_slice
        
        return image

    def get_label(self, item):
        item['volume'] = item['volume'].replace('seismic', 'horizon_labels')
        label = self.get_slice(item, dtype=torch.uint8)

        return label

    def process_label(self, label):
        label_type = self.wandb_config['label_type']

        # if label_type == 'border':
        #     label = self.get_border_label(label)
        # elif label_type == 'layer':
        #     label = self.get_layer_label(label)
        # elif label_type == 'instance':
        if label_type == 'instance':
            label = self.get_instance_label(label)
        # elif label_type == 'semantic':
        #     label = label
        else:
            raise ValueError(f'Unknown label_type: {label_type}')

        return label

    # def get_border_label(self, label, kernel=3):
    #     label = label.view(1, 1, label.shape[0], label.shape[1]).float()
    #     pad_size = (kernel - 1) // 2
    #     padded_label = tF.pad(label, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    #     unfolded = padded_label.unfold(2, kernel, 1).unfold(3, kernel, 1)
    #     binary_label = (unfolded.std(dim=(4, 5)) == 0).byte()
    #     binary_label = 1 - binary_label.squeeze()

        return binary_label

    # def get_layer_label(self, label):
    #     binary_label = torch.where(label % 2 == 0, 1, 0)

    #     return binary_label

    def get_instance_label(self, label):
        label = label - label.min()

        return label

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

if __name__ == '__main__':
    import src.models.mask2former.make_lightning as ml
    from torch.utils.data import DataLoader

    config = utils.get_config()
    wandb_config = utils.init_wandb('mask2former.yml')

    processor = ml.get_processor(config, wandb_config)
    
    train_volumes = get_volumes(config, set='train')

    args = {
        'config': config,
        'wandb_config': wandb_config,
        'processor': processor,
        'volumes': train_volumes,
    }

    train_dataset = SegSubDataset(args)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb_config['batch_size'],
        shuffle=False
    )

    for item, inputs in train_dataloader:
        break
