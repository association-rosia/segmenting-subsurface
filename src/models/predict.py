import os
import random
import shutil
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as tF
import torchvision.transforms.functional as tvF
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.make_dataset as md
import src.utils as utils

warnings.filterwarnings('ignore')


def main():
    mask2former_id = '3xg8r6lz'
    segment_anything_id = '2snz8a1d'

    config = utils.get_config()
    submission_path = create_path(config, mask2former_id, segment_anything_id)

    m2f_run = get_run(mask2former_id)
    m2f_lightning, m2f_processor = load_model_processor(config, m2f_run, 'mask2former')

    sam_run = get_run(segment_anything_id)
    sam_lightning, sam_processor = load_model_processor(config, sam_run, 'segment_anything')

    test_volumes = md.get_volumes(config, set='test')

    args = {
        'config': config,
        'wandb_config': m2f_run.config,
        'processor': m2f_processor,
        'volumes': test_volumes,
        'set': 'test'
    }

    test_dataset = md.SegSubDataset(args)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=300,
        shuffle=False
    )

    with torch.no_grad():
        for item, inputs in tqdm(test_dataloader):
            save_path = get_save_path(item, submission_path)
            m2f_inputs = preprocess(inputs)
            # m2f_outputs = predict_mask2former(m2f_lightning, m2f_processor, m2f_inputs)
            m2f_outputs = get_m2f_outputs_example()
            sam_input_points = create_sam_input_points(m2f_outputs, m2f_inputs, sam_run)
            sam_outputs = predict_segment_anything(sam_lightning, m2f_inputs, sam_input_points)

            outputs = unprocess(outputs)
            save_outputs(outputs, save_path)

    shutil.make_archive(submission_path, 'zip', submission_path)


def create_sam_input_points(m2f_outputs, m2f_inputs, sam_run):
    sam_input_points = []

    # add multiprocessing
    for i in tqdm(range(m2f_outputs.shape[0])):
        m2f_output = m2f_outputs[i]
        indexes = torch.unique(m2f_output).tolist()

        if len(indexes) == 1:
            utils.plot_slice(m2f_inputs['pixel_values'][i])
            utils.plot_slice(m2f_outputs[i])
            print()

        m2f_output = tF.one_hot(m2f_output.to(torch.int64)).to(torch.uint8)
        m2f_output = torch.permute(m2f_output, (2, 0, 1))
        m2f_output = m2f_output[indexes]
        m2f_output = utils.resize_tensor_2d(m2f_output, (1024, 1024))

        input_points = []
        for i in range(m2f_output.shape[0]):
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(12, 12))
            opened_m2f_output_i = cv2.morphologyEx(m2f_output[i].numpy(), cv2.MORPH_OPEN, kernel=kernel)
            opened_m2f_output_i = torch.from_numpy(opened_m2f_output_i)

            valid_label = 1 in torch.unique(opened_m2f_output_i).tolist()
            if valid_label:
                # utils.plot_slice(opened_m2f_output_i)
                input_points_coord = torch.argwhere(opened_m2f_output_i).tolist()
                input_points_coord = random.choices(input_points_coord, k=sam_run.config['num_input_points'])
                input_points.append(torch.tensor(input_points_coord).unsqueeze(0))

        sam_input_points.append(input_points)

    return sam_input_points


def predict_mask2former(m2f_lightning, m2f_processor, m2f_inputs):
    outputs = m2f_lightning(m2f_inputs)
    outputs = m2f_processor.post_process_instance_segmentation(outputs)
    outputs = torch.stack([slice['segmentation'] for slice in outputs])

    return outputs


def predict_segment_anything(sam_lightning, m2f_inputs, sam_input_points):
    outputs = None
    return outputs


def preprocess(inputs):
    device = utils.get_device()
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
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
    outputs = tF.interpolate(outputs.unsqueeze(0), size=(100, 300), mode='bilinear', align_corners=False)
    outputs = outputs.squeeze(0)
    outputs = torch.movedim(outputs, 1, 2)
    outputs = outputs.to(torch.uint8)

    return outputs


def save_outputs(outputs, save_path):
    outputs = outputs.detach().cpu().numpy()
    np.save(save_path, outputs, allow_pickle=True)


def load_model_processor(config, run, model_name):
    lightning = None

    if model_name == 'mask2former':
        lightning = utils.load_mask2former(config, run)
    elif model_name == 'segment_anything':
        lightning = utils.load_segment_anything(config, run)

    processor = lightning.processor

    return lightning, processor


def get_run(run_id):
    run = utils.get_run(run_id)
    run.config['dilation'] = 1

    return run


def create_path(config, mask2former_id, segment_anything_id):
    submission_path = os.path.join(config['path']['submissions']['root'], f'{mask2former_id}_{segment_anything_id}')
    os.makedirs(submission_path, exist_ok=True)

    return submission_path


def get_m2f_outputs_example():
    m2f_outputs = torch.from_numpy(np.load('data/processed/sub_vol_40.npy', allow_pickle=True))
    m2f_outputs = torch.movedim(m2f_outputs, 2, 1)
    m2f_outputs = tvF.resize(m2f_outputs, size=(384, 384), interpolation=tvF.InterpolationMode.NEAREST_EXACT)

    return m2f_outputs


if __name__ == '__main__':
    main()
