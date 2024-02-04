import multiprocessing as mp
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
            m2f_inputs, m2f_outputs = get_m2f_outputs_example(config, item, m2f_inputs)
            print('m2f_inputs, m2f_outputs = get_m2f_outputs_example(config, item, m2f_inputs)')
            sam_input_points, sam_input_points_stack_num = create_sam_input_points(m2f_outputs, item, sam_run)
            print('sam_input_points, sam_input_points_stack_num = create_sam_input_points(m2f_outputs, item, sam_run)')
            sam_outputs = predict_segment_anything(sam_lightning, m2f_inputs, m2f_outputs, sam_input_points,
                                                   sam_input_points_stack_num)
            print('sam_outputs = predict_segment_anything(sam_lightning, m2f_inputs, m2f_outputs, sam_input_points, '
                  'sam_input_points_stack_num)')
            outputs = unprocess(sam_outputs)
            print('outputs = unprocess(sam_outputs)')
            save_outputs(outputs, save_path)
            print('save_outputs(outputs, save_path)')

    shutil.make_archive(submission_path, 'zip', submission_path)


def split_list_args(args, nb_split):
    sub_len = len(args) // nb_split
    list_args_split = [args[sub_len * i:sub_len * (i + 1)] for i in range(nb_split - 1)]
    list_args_split.append(args[sub_len * (nb_split - 1):])

    return list_args_split


def create_sam_input_points(m2f_outputs, item, sam_run):
    mp.set_start_method('spawn', force=True)
    m2f_outputs.share_memory_()
    manager = mp.Manager()
    sam_input_points = manager.list()

    volumes = item['volume']
    slices = item['slice']

    m2f_args = [(m2f_outputs[i].cpu(), volumes[i], slices[i].item(), sam_run.config) for i in range(len(m2f_outputs))]
    list_args_split = split_list_args(m2f_args, nb_split=torch.cuda.device_count())

    list_process = [
        mp.Process(target=extract_input_points(args_split, sam_input_points))
        for i, args_split in enumerate(list_args_split)
    ]

    for p in list_process:
        p.start()

    for p in list_process:
        p.join()

    max_input_points = max([len(input_points) for input_points in sam_input_points])
    sam_input_points_stack_num = [max_input_points - len(sam_input_points[i]) for i in range(len(sam_input_points))]

    sam_input_points_stack = [sam_input_points[i][0].unsqueeze(0).repeat(sam_input_points_stack_num[i], 1, 1) for i in
                              range(len(sam_input_points))]

    sam_input_points = [torch.cat([sam_input_points[i], sam_input_points_stack[i]]) for i in
                        range(len(sam_input_points))]

    sam_input_points = torch.stack(sam_input_points)

    return sam_input_points, sam_input_points_stack_num


def extract_input_points(args_split, sam_input_points):
    for m2f_output, volume, slice, sam_config in args_split:
        indexes = torch.unique(m2f_output).tolist()

        if len(indexes) == 1:
            print(volume, slice)

        m2f_output = tF.one_hot(m2f_output.to(torch.int64)).to(torch.uint8)
        m2f_output = torch.permute(m2f_output, (2, 0, 1))
        m2f_output = m2f_output[indexes]
        m2f_output = utils.resize_tensor_2d(m2f_output, (1024, 1024))

        input_points = []
        for i in range(m2f_output.shape[0]):
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(12, 12))
            opened_m2f_output_i = cv2.morphologyEx(m2f_output[i].cpu().numpy(), cv2.MORPH_OPEN, kernel=kernel)
            opened_m2f_output_i = torch.from_numpy(opened_m2f_output_i)

            valid_label = 1 in torch.unique(opened_m2f_output_i).tolist()

            if valid_label:
                count_1 = torch.unique(opened_m2f_output_i, return_counts=True)[1][1].item()

                if count_1 > 1000:
                    input_points_argw = torch.argwhere(opened_m2f_output_i)
                    input_points_idx = random.sample(range(len(input_points_argw)), k=sam_config['num_input_points'])
                    input_points_coord = input_points_argw[input_points_idx]
                    input_points.append(input_points_coord)

        device = utils.get_device()
        input_points = torch.stack(input_points).to(device)
        sam_input_points.append(input_points)

    # return input_points_split


def predict_mask2former(m2f_lightning, m2f_processor, m2f_inputs):
    outputs = m2f_lightning(m2f_inputs)
    outputs = m2f_processor.post_process_instance_segmentation(outputs)
    outputs = torch.stack([slice['segmentation'] for slice in outputs])

    return outputs


def predict_segment_anything(sam_lightning, m2f_inputs, m2f_outputs, sam_input_points, sam_input_points_stack_num,
                             iou_threshold=0):
    filtered_sam_outputs = []
    sam_pixel_values = tvF.resize(m2f_inputs['pixel_values'], (1024, 1024))

    sam_outputs = sam_lightning.model(
        pixel_values=sam_pixel_values,
        input_points=sam_input_points,
        multimask_output=False
    )

    for i in range(len(sam_outputs.pred_masks)):
        pred_masks = sam_outputs.pred_masks[i].squeeze()
        pred_masks = pred_masks[:(len(pred_masks) - sam_input_points_stack_num[i])]
        iou_scores = sam_outputs.iou_scores[i].squeeze().tolist()
        iou_scores = iou_scores[:(len(iou_scores) - sam_input_points_stack_num[i])]
        filtered_iou_scores_idx = [i for i, score in enumerate(iou_scores) if score > iou_threshold]
        filtered_pred_masks = pred_masks[filtered_iou_scores_idx]
        filtered_outputs = (tF.sigmoid(filtered_pred_masks).argmax(dim=0)).type(torch.uint8)
        filtered_sam_outputs.append(filtered_outputs)
        # utils.plot_slice(filtered_outputs)

    filtered_sam_outputs = torch.stack(filtered_sam_outputs)

    return filtered_sam_outputs


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
    outputs = outputs.unsqueeze(0).to(torch.float16)
    outputs = tF.interpolate(outputs, size=(100, 300), mode='bilinear', align_corners=False)
    outputs = outputs.squeeze(0).to(torch.uint8)
    outputs = torch.movedim(outputs, 1, 2)

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


def get_m2f_outputs_example(config, item, m2f_inputs):
    device = utils.get_device()
    file_name = item['volume'][0].split('/')[-1].replace('test', 'sub')
    path = os.path.join(config['path']['data']['processed']['test'], 'stellar-durian-37-3xg8r6lz', file_name)
    m2f_outputs = torch.from_numpy(np.load(path, allow_pickle=True)).to(device)
    m2f_outputs = torch.movedim(m2f_outputs, 2, 1)
    m2f_outputs = tvF.resize(m2f_outputs, size=(384, 384), interpolation=tvF.InterpolationMode.NEAREST_EXACT)

    m2f_inputs['pixel_values'] = m2f_inputs['pixel_values'][:5]
    m2f_outputs = m2f_outputs[:5]

    return m2f_inputs, m2f_outputs


if __name__ == '__main__':
    main()
