import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as tF
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
    mask2former_id = None  # '4u9c0boz'
    segment_anything_id = 'wgeuew2w'

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

    with torch.no_grad():
        for item, inputs in tqdm(test_dataloader):
            save_path = get_save_path(item, submission_path)
            inputs = preprocess(inputs)

            # inputs_prompts =

            outputs = predict_sam(model, processor, inputs)
            # outputs = predict(model, item, inputs)
            outputs = unprocess(outputs)
            save_outputs(outputs, save_path)

    shutil.make_archive(submission_path, 'zip', submission_path)


def predict_sam(model, processor, inputs):
    pixel_values = inputs['pixel_values']

    for idx in range(pixel_values.shape[0]):
        slice = pixel_values[idx]
        model_input = preprocess_sam(processor, slice)
        model_output = model.model(**model_input)
        print()

    return inputs


def preprocess_sam(processor, slice, points_per_batch=64):
    target_size = processor.size['longest_edge']
    crop_boxes, grid_points, cropped_images, input_labels = processor.generate_crop_boxes(slice, target_size)
    n_points = grid_points.shape[1]
    batched_points_list = []
    input_labels_list = []
    input_boxes_list = []
    is_last_list = []
    pixel_values_list = []

    for i in range(0, n_points, points_per_batch):
        batched_points = grid_points[:, i: i + points_per_batch, :, :].squeeze(0)
        labels = input_labels[:, i: i + points_per_batch].squeeze(0)
        is_last = i == n_points - points_per_batch

        batched_points_list.append(batched_points)
        input_labels_list.append(labels)
        input_boxes_list.append(crop_boxes)
        is_last_list.append(is_last)
        pixel_values_list.append(slice)

    return {
        'input_points': torch.stack(batched_points_list, dim=0),
        'input_labels': torch.stack(input_labels_list).squeeze(),
        'input_boxes': torch.stack(input_boxes_list).squeeze(),
        'is_last': is_last_list,
        'pixel_values': torch.stack(pixel_values_list),
    }


# def predict(model, item, inputs, n=10):
#     device = utils.get_device()
#     inputs_shape = inputs['pixel_values'].shape
#     outputs = torch.zeros((inputs_shape[0], inputs_shape[-2], inputs_shape[-1]), dtype=torch.uint8, device=device)
#     indexes = item['slice'].view(n, 300 // n).tolist()
#
#     for index in indexes:
#         inputs_index = dict()
#         inputs_index['pixel_values'] = inputs['pixel_values'][index]
#         outputs_index = model(inputs_index)
#
#         if outputs_index.dim() == 4:
#             outputs[index] = (tF.sigmoid(outputs_index).argmax(dim=1)).type(torch.uint8)
#         else:
#             outputs[index] = outputs_index.type(torch.uint8)
#
#     return outputs


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
