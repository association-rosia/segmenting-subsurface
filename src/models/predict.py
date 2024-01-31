import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.make_dataset as md
import src.utils as utils

warnings.filterwarnings('ignore')


class MokeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return (torch.max(inputs['pixel_values'], dim=1).values > 0).to(torch.float32)


def main():
    segformer_id = '4u9c0boz'
    mask2former_id = segformer_id
    segment_anything_id = None

    config = utils.get_config()
    submission_path = create_path(config, mask2former_id, segment_anything_id)

    run = get_run_from_model_id(mask2former_id, segment_anything_id)
    model, processor = load_model_processor(config, run, mask2former_id, segment_anything_id)
    test_volumes = md.get_volumes(config, set='test')

    args = {
        'config': config,
        'wandb_config': run.config,
        'processor': processor,
        'volumes': test_volumes,
        'set': 'test'
    }

    test_dataset = md.SegSubDataset(args)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=300,
        shuffle=False
    )

    # model = MokeModel()

    for item, inputs in tqdm(test_dataloader):
        save_path = get_save_path(item, submission_path)
        inputs = send_inputs_to_device(inputs)
        outputs = model(inputs)
        outputs = unprocess(outputs)
        save_outputs(outputs, save_path)

    shutil.make_archive(submission_path, 'zip', submission_path)


def send_inputs_to_device(inputs):
    device = utils.get_device()
    inputs['pixel_values'] = inputs['pixel_values'].to(device)

    return inputs


def get_save_path(item, submission_path):
    volume_path = list(set(item['volume']))
    assert len(volume_path) == 1
    volume_path = volume_path[0]
    volume_path = os.path.split(volume_path)[-1]
    volume_path = volume_path.replace('test', 'sub')
    save_path = os.path.join(submission_path, volume_path)

    return save_path


def unprocess(outputs):
    outputs = F.interpolate(outputs.unsqueeze(0), size=(100, 300), mode='bilinear', align_corners=False)
    outputs = outputs.squeeze(0)
    outputs = torch.movedim(outputs, 1, 2)
    outputs = outputs.to(torch.int8)

    return outputs


def save_outputs(outputs, save_path):
    outputs = outputs.detach().cpu().numpy()

    with open(save_path, 'wb') as f:
        np.save(f, outputs)


def load_model_processor(config, run, mask2former_id, segment_anything_id):
    lightning = None

    if mask2former_id and segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both defined: {mask2former_id} - {segment_anything_id}")
    elif not mask2former_id and not segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both null")
    elif mask2former_id:
        lightning = utils.load_segformer_model(config, run)
    else:
        lightning = utils.load_segment_anything(config, run)

    processor = lightning.processor

    return lightning, processor


def get_run_from_model_id(mask2former_id, segment_anything_id):
    run = None

    if mask2former_id and segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both defined: {mask2former_id} - {segment_anything_id}")
    elif not mask2former_id and not segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both null")
    elif mask2former_id:
        run = utils.get_run(mask2former_id)
    else:
        run = utils.get_run(segment_anything_id)

    run.config['dilation'] = 1

    return run


def create_path(config, mask2former_id, segment_anything_id):
    submission_path = os.path.join(config['path']['submissions']['root'], f'{mask2former_id}_{segment_anything_id}')
    os.makedirs(submission_path, exist_ok=True)

    return submission_path


if __name__ == '__main__':
    main()
