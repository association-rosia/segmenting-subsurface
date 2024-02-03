import os
import sys

sys.path.append(os.curdir)

from glob import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
        self.set = args['set']
        self.slices = self.get_slices()
        self.volume_min = self.config['data']['min']
        self.volume_max = self.config['data']['max']

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item = self.slices[idx]
        image = self.get_image(item)
        label = self.get_label(item)

        if self.wandb_config['label_type'] == 'instance':
            inputs = self.create_mask2former_inputs(image, label)
        else:
            inputs = self.processor(
                images=image,
                segmentation_maps=label,
                return_tensors='pt'
            )

            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

            if 'reshaped_input_sizes' in inputs:
                inputs = self.create_sam_inputs(inputs, label)
            else:
                inputs['labels'] = self.process_label(inputs['labels'])

        return item, inputs

    def create_sam_inputs(self, inputs, label):
        inputs['pixel_values'] = tvF.resize(inputs['pixel_values'], (1024, 1024))

        if self.set == 'train':
            inputs['labels'] = utils.resize_tensor_2d(label, (1024, 1024))
            inputs['labels'] = inputs['labels'][random.randint(0, inputs['labels'].shape[0] - 1)]
            input_points_coord = torch.argwhere(inputs['labels']).tolist()
            input_points_coord = random.choices(input_points_coord, k=self.wandb_config['num_input_points'])
            inputs['input_points'] = torch.tensor(input_points_coord).unsqueeze(0)
            inputs['labels'] = utils.resize_tensor_2d(inputs['labels'], (256, 256))

        return inputs

    def create_mask2former_inputs(self, image, label):
        if self.set == 'train':
            label = self.process_label(label)
            instance_id_to_semantic_id = {int(i): 0 for i in np.unique(label)}
        else:
            instance_id_to_semantic_id = None

        inputs = self.processor(
            images=image,
            segmentation_maps=label,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors='pt'
        )

        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

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

    def build_image(self, slice, segformer_mask):
        num_channels_mask = self.wandb_config.get('num_channels_mask')

        image = None
        if not num_channels_mask or num_channels_mask == 0:
            image = torch.stack([slice, slice, slice])
        elif num_channels_mask == 1:
            image = torch.stack([slice, segformer_mask, slice])
        elif num_channels_mask == 2:
            image = torch.stack([segformer_mask, slice, segformer_mask])
        elif num_channels_mask == 3:
            image = torch.stack([segformer_mask, segformer_mask, segformer_mask])
        else:
            ValueError(f'Wrong num_channels_mask value: {num_channels_mask}')

        return image

    def get_segformer_mask(self, item):
        segformer_mask = None

        if self.wandb_config.get('segformer_id') and self.wandb_config['num_channels_mask'] > 0:
            data_path = self.config['path']['data']['processed'][self.set]

            volume_file = item['volume'].split('/')[-1]
            if self.set == 'train':
                volume_file = volume_file.replace('seismic', 'binary_mask')
            else:
                volume_file = volume_file.replace('vol', 'bmask')

            path = os.path.join(data_path, self.wandb_config['segformer_id'], volume_file)
            segformer_mask = self.get_slice(path, item, dtype=torch.float32)

        return segformer_mask

    def get_slice(self, path, item, dtype, type=None):
        volume = np.load(path, allow_pickle=True)

        if item['dim'] == '0':
            slice = volume[item['slice'], :, :]
        elif item['dim'] == '1':
            slice = volume[:, item['slice'], :]
        else:
            raise ValueError(f'Unknown dimension: {item["dim"]}')

        slice = torch.from_numpy(slice)
        slice = slice.to(dtype=dtype)
        slice = torch.movedim(slice, 0, 1)

        if type == 'pixel_values':
            slice = self.scale(slice).unsqueeze(0)
            slice = tvF.adjust_contrast(slice, contrast_factor=self.wandb_config['contrast_factor'])
            slice = slice.squeeze()

        return slice

    def scale(self, image):
        image = (image - self.volume_min) / (self.volume_max - self.volume_min)

        return image

    def get_image(self, item):
        path = item['volume']
        slice = self.get_slice(path, item, dtype=torch.float32, type='pixel_values')
        segformer_mask = self.get_segformer_mask(item)
        image = self.build_image(slice, segformer_mask)

        return image

    def get_label(self, item):
        label = None
        if self.set != 'test':
            path = item['volume'].replace('seismic', 'horizon_labels')
            label = self.get_slice(path, item, dtype=torch.uint8)

        return label

    def process_label(self, label):
        label_type = self.wandb_config['label_type']

        if label_type == 'border':
            label = self.get_border_label(label)
        elif label_type == 'layer':
            label = self.get_layer_label(label)
        elif label_type == 'instance':
            label = self.get_instance_label(label)
        elif label_type == 'one_hot':
            label = self.get_one_hot_label(label)
        elif label_type == 'semantic':
            label = label
        else:
            raise ValueError(f'Unknown label_type: {label_type}')

        return label

    def get_one_hot_label(self, label):
        indexes = torch.unique(label)
        label = torch.permute(tF.one_hot(label.to(torch.int64)), (2, 0, 1))
        label = label[indexes.tolist()].to(torch.uint8)

        return label

    def get_border_label(self, label, kernel=3):
        label = label.view(1, 1, label.shape[0], label.shape[1]).float()
        pad_size = (kernel - 1) // 2
        padded_label = tF.pad(label, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        unfolded = padded_label.unfold(2, kernel, 1).unfold(3, kernel, 1)
        binary_label = (unfolded.std(dim=(4, 5)) == 0).byte()
        binary_label = 1 - binary_label.squeeze()

        return binary_label

    def get_layer_label(self, label):
        binary_label = torch.where(label % 2 == 0, 1, 0)

        return binary_label

    def get_instance_label(self, label):
        instance_label = np.full(label.shape, np.nan)
        old_labels = np.unique(label)
        new_labels = range(len(old_labels))
        for old_label, new_label in zip(old_labels, new_labels):
            instance_label = np.where(label == old_label, new_label, instance_label)

        return instance_label


def collate_fn(batch):
    pixel_values = torch.stack([el[1]['pixel_values'] for el in batch])
    pixel_mask = torch.stack([el[1]['pixel_mask'] for el in batch])
    class_labels = [el[1]['class_labels'] for el in batch]
    mask_labels = [el[1]['mask_labels'] for el in batch]
    slices = [el[0] for el in batch]

    inputs = {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'class_labels': class_labels,
        'mask_labels': mask_labels
    }

    return slices, inputs


def get_volumes(config, set):
    root_test = config['path']['data']['raw']['test']
    root_train = config['path']['data']['raw']['train']
    notebooks_test = os.path.join(os.pardir, root_test)
    notebooks_train = os.path.join(os.pardir, root_train)

    root = root_test if set == 'test' else root_train
    notebooks = notebooks_test if set == 'test' else notebooks_train
    path = root if os.path.exists(root) else notebooks

    search_volumes = os.path.join(path, '**', '*.npy')
    volumes = glob(search_volumes, recursive=True)
    volumes = [volume for volume in volumes if 'seismic_block' in volume or set == 'test']

    return volumes


def get_training_volumes(config, wandb_config):
    training_volumes = get_volumes(config, set='training')
    train_volumes, val_volumes = train_test_split(training_volumes,
                                                  test_size=wandb_config['val_size'],
                                                  random_state=wandb_config['random_state'])

    return train_volumes, val_volumes


def get_class_frequencies(train_dataloader):
    class_frequencies = {}
    max_num_classes = 0
    count_all = 0

    for _, inputs in tqdm(train_dataloader):
        labels = inputs['labels']
        values, counts = labels.unique(return_counts=True)
        max_num_classes = max(max_num_classes, len(values))
        count_all += counts.sum().item()

        for value, count in zip(values, counts):
            value = value.item()
            count = count.item()

            if value in class_frequencies.keys():
                class_frequencies[value] += count
            else:
                class_frequencies[value] = count

    class_weights = {k: 1 / (v / count_all) for k, v in class_frequencies.items()}
    class_weights_list = [v for _, v in class_weights.items()]
    class_weights_proba = [el / sum(class_weights_list) for el in class_weights_list]

    print('\nclass_weights', class_weights)
    print('\nclass_weights_list', class_weights_list)
    print('\nclass_weights_proba', class_weights_proba)
    print('\nnum_labels', len(class_weights_proba), '- max_num_classes', max_num_classes)
    pass


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
    import src.models.segment_anything.make_lightning as sam_ml
    # import src.models.segformer.make_lightning as seg_ml
    from torch.utils.data import DataLoader

    config = utils.get_config()
    # compute_image_mean_std(config)
    wandb_config = utils.init_wandb('mask2former.yml')

    model = sam_ml.get_model(wandb_config)
    processor = utils.get_processor(config, wandb_config)
    train_volumes = get_volumes(config, set='train')

    args = {
        'config': config,
        'wandb_config': wandb_config,
        'processor': processor,
        'volumes': train_volumes,
        'set': 'train'
    }

    train_dataset = SegSubDataset(args)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # get_class_frequencies(train_dataloader)

    for item, inputs in train_dataloader:
        break
