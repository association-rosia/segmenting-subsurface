import os
import sys

sys.path.append(os.curdir)

import wandb
import torch
import torch.nn.functional as tF
import torchmetrics.functional as tmF
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torchmetrics as tm
from sklearn.metrics import pairwise_distances

import src.models.losses as losses
import src.data.make_dataset as md
from src import utils


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.config = args['config']
        self.wandb = args['wandb']
        self.model = args['model']
        self.processor = args['processor']
        self.train_volumes = args['train_volumes']
        self.val_volumes = args['val_volumes']

        self.criterion = self.configure_criterion()
        self.metrics = self.configure_metrics()

    def forward(self, inputs):
        pixel_values = inputs['pixel_values']
        outputs = self.model(pixel_values=pixel_values)

        upsampled_logits = tF.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        upsampled_logits = upsampled_logits.squeeze(1)

        return upsampled_logits

    def training_step(self, batch):
        item, inputs = batch
        outputs = self.forward(inputs)
        outputs = self.reorder(outputs, inputs['labels'])
        loss = self.criterion(outputs, inputs['labels'])
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        item, inputs = batch
        outputs = self.forward(inputs)
        outputs = self.reorder(outputs, inputs['labels'])
        loss = self.criterion(outputs, inputs['labels'])
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        self.validation_log(batch_idx, outputs, inputs)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.metrics.reset()

    def reorder(self, outputs, labels):

        def dice(labels, outputs):
            labels = torch.from_numpy(labels).to(dtype=torch.int64)
            outputs = torch.from_numpy(outputs).to(dtype=torch.int64)

            return tmF.dice(labels, outputs)

        if self.wandb.config.label_reorder:
            outputs = self.logits_to_labels(outputs)

            for b in range(outputs.shape[0]):
                num_classes = self.wandb.config.num_labels
                label = torch.permute(tF.one_hot(labels[b].to(torch.int64), num_classes=num_classes), (2, 0, 1))
                flatten_label = torch.flatten(label, start_dim=1, end_dim=2)
                output = torch.permute(tF.one_hot(outputs[b].to(torch.int64), num_classes=num_classes), (2, 0, 1))
                flatten_output = torch.flatten(output, start_dim=1, end_dim=2)

                distances = torch.from_numpy(pairwise_distances(flatten_label.cpu(), flatten_output.cpu(), metric=dice))
                print(distances)
                labels_indexes = [i for i, v in enumerate(flatten_label.sum(dim=1).tolist()) if v != 0]
                print(labels_indexes)

                indexes_reordered = []
                for index in labels_indexes:
                    print('index', index)
                    is_matched = False
                    distance = distances[index]
                    print('distance', distance)

                    while not is_matched:
                        distance_argmax = distance.argmax().item()
                        print('distance_argmax', distance_argmax)
                        max_row = distance.max().item()
                        max_col = distances[:, distance_argmax].max().item()
                        is_empty = flatten_output[distance_argmax].sum().item() == 0

                        if (max_row == max_col or is_empty) and distance_argmax not in indexes_reordered:
                            is_matched = True
                            indexes_reordered.append(distance_argmax)
                        else:
                            is_matched = False
                            distances[distance_argmax] = -1

                indexes_reordered += [i for i in range(num_classes) if i not in indexes_reordered]
                outputs[b] = output[indexes_reordered]

        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb.config.lr)

        return optimizer

    def logits_to_labels(self, outputs):
        num_labels = self.wandb.config.num_labels

        if num_labels == 1:
            outputs = (tF.sigmoid(outputs) > 0.5).type(torch.uint8)
        elif num_labels > 1:
            outputs = (tF.sigmoid(outputs).argmax(dim=1)).type(torch.uint8)
        else:
            raise ValueError(f'Invalid num_labels: {self.num_labels}')

        return outputs

    def validation_log(self, batch_idx, outputs, inputs):
        outputs = self.logits_to_labels(outputs)
        self.metrics.update(outputs, inputs['labels'])

        if batch_idx == 0:
            self.log_image(inputs, outputs)

    def configure_criterion(self):
        class_weights = self.get_class_weights()
        num_labels = self.wandb.config.num_labels

        if self.wandb.config.criterion == 'CrossEntropyLoss':
            criterion = losses.CrossEntropyLoss(num_labels=num_labels, class_weights=class_weights)
        elif self.wandb.config.criterion == 'DiceLoss':
            criterion = losses.DiceLoss(num_labels=num_labels)
        elif self.wandb.config.criterion == 'DiceCrossEntropyLoss':
            criterion = losses.DiceCrossEntropyLoss(num_labels=num_labels, class_weights=class_weights)
        elif self.wandb.config.criterion == 'JaccardCrossEntropyLoss':
            criterion = losses.JaccardCrossEntropyLoss(num_labels=num_labels, class_weights=class_weights)
        else:
            raise ValueError(f'Unknown criterion: {self.wandb.config.criterion}')

        return criterion

    def configure_metrics(self):
        num_labels = self.wandb.config.num_labels

        if num_labels == 1:
            metrics = tm.MetricCollection({
                'val/dice': tm.Dice(),
                'val/iou': tm.classification.BinaryJaccardIndex()
            })
        elif num_labels > 1:
            metrics = tm.MetricCollection({
                'val/dice': tm.Dice(num_classes=num_labels, average='macro'),
                'val/iou': tm.JaccardIndex(task='multiclass', num_classes=num_labels),
            })
        else:
            raise ValueError(f'Invalid num_labels: {num_labels}')

        return metrics

    def log_image(self, inputs, outputs):
        pixel_values = inputs['pixel_values'][0][0].numpy(force=True)
        outputs = outputs[0].numpy(force=True)
        ground_truth = inputs['labels'][0].numpy(force=True)

        self.wandb.log(
            {'val/prediction': wandb.Image(pixel_values, masks={
                'predictions': {
                    'mask_data': outputs
                },
                'ground_truth': {
                    'mask_data': ground_truth
                }
            })})

    def get_class_weights(self):
        label_type = self.wandb.config.label_type

        if label_type == 'border':
            class_weights = torch.Tensor([15])
        elif label_type == 'layer':
            class_weights = None
        elif label_type == 'semantic':
            class_weights = None
        elif label_type == 'instance':
            class_weights = None
        else:
            raise ValueError(f'Unknown label_type: {label_type}')

        return class_weights

    def train_dataloader(self):
        args = {
            'config': self.config,
            'wandb': self.wandb,
            'processor': self.processor,
            'volumes': self.train_volumes
        }

        dataset_train = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.wandb.config.batch_size,
            num_workers=self.wandb.config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        args = {
            'config': self.config,
            'wandb': self.wandb,
            'processor': self.processor,
            'volumes': self.val_volumes
        }

        dataset_val = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.wandb.config.batch_size,
            num_workers=self.wandb.config.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )


def get_processor_model(config, wandb):
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        do_rescale=False,
        image_mean=config['data']['mean'],
        image_std=config['data']['std'],
        do_reduce_labels=False
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        num_labels=wandb.config.num_labels,
        num_channels=wandb.config.num_channels,
        ignore_mismatched_sizes=True
    )

    return processor, model


if __name__ == '__main__':
    config = utils.get_config()
    wandb = utils.init_wandb()
    processor, model = get_processor_model(config, wandb)
    train_volumes, val_volumes = md.get_training_volumes(config, wandb)

    args = {
        'config': config,
        'wandb': wandb,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }

    lightning = SegSubLightning(args)

    pass
