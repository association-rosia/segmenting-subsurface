import os
import sys

sys.path.append(os.curdir)

from torch import nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torchmetrics as tm

import src.data.make_dataset as md
import utils


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.config = args['config']
        self.wandb = args['wandb']
        self.model = args['model']
        self.processor = args['processor']
        self.train_volumes = args['train_volumes']
        self.val_volumes = args['val_volumes']

        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = tm.MetricCollection({
            'val/iou': tm.classification.BinaryJaccardIndex(),
            'val/acc': tm.classification.BinaryAccuracy(),
            'val/dice': tm.classification.Dice()
        })

    def forward(self, inputs):
        pixel_values = inputs['pixel_values']
        outputs = self.model(pixel_values=pixel_values)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False
        )

        return upsampled_logits

    def training_step(self, batch):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, inputs['labels'])
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, inputs['labels'])
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        print(outputs.shape, inputs['labels'].shape)

        self.metrics.update(outputs, inputs['labels'])

        if batch_idx == 0:
            self.log_image(inputs, outputs)

        return loss

    def log_image(self, inputs, outputs):
        pixel_values = inputs['pixel_values'][0][0].numpy()
        predictions = outputs[0].numpy()
        ground_truth = inputs['labels'][0].numpy()

        self.logger.experiment.log({
            'pixel_values': wandb.Image(
                pixel_values,
                masks={
                    'predictions': {
                        'mask_data': predictions
                    },
                    'ground_truth': {
                        'mask_data': ground_truth
                    }
                }
            )
        })

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb.config.lr)

        return optimizer

    def train_dataloader(self):
        args = {
            'config': self.config,
            'wandb': self.wandb,
            'processor': self.processor,
            'volumes': self.train_volumes,
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
            'volumes': self.val_volumes,
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
        do_reduce_labels=False,
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
