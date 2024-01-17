import os
import sys

sys.path.append(os.curdir)

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torchmetrics.classification import Dice

import src.data.make_dataset as md
import utils


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.config = args['config']
        self.wandb = args['wandb']
        self.model = args['model']
        self.processor = args['processor']
        self.train_slices = args['train_slices']
        self.val_slices = args['val_slices']
        # self.val_dice = Dice(num_classes=1, threshold=0.8, average='macro')

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, batch_size=self.wandb.config.batch_size, sync_dist=True)

        return loss

    # def log_slice_mask(self, image, mask_target, mask_pred):
    #     pass

    def validation_step(self, batch, batch_idx):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        # outputs = self.processor.post_process_instance_segmentation(outputs)
        self.log('val/loss', loss, on_step=True, batch_size=self.wandb.config.batch_size, sync_dist=True)
        # self.val_dice.update(logits, y)

        return loss

    # def on_validation_epoch_end(self):
    #     self.log('val/dice', self.val_dice.compute(), on_epoch=True, sync_dist=True)
    #     self.val_dice.reset()

    # def test_step(self, batch, batch_idx):
    #     pass

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.wandb.config.lr,
            weight_decay=self.wandb.config.weight_decay
        )

        return optimizer

    def train_dataloader(self):
        args = {
            'config': self.config,
            'wandb': self.wandb,
            'processor': self.processor,
            'slices': self.train_slices,
        }

        dataset_train = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.wandb.config.batch_size,
            num_workers=self.wandb.config.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=md.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        args = {
            'config': self.config,
            'wandb': self.wandb,
            'processor': self.processor,
            'slices': self.val_slices,
        }

        dataset_val = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.wandb.config.batch_size,
            num_workers=self.wandb.config.num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=md.collate_fn,
            pin_memory=True
        )


def get_processor_model(config, wandb):
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        do_rescale=False,
        image_mean=config['data']['mean'],
        image_std=config['data']['std'],
        num_labels=wandb.config.num_labels
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        num_labels=wandb.config.num_labels,
        # backbone_config={'num_channels': wandb.config.num_channels},
        ignore_mismatched_sizes=True
    )

    return processor, model


if __name__ == '__main__':
    config = utils.get_config()
    wandb = utils.init_wandb()
    processor, model = get_processor_model(config, wandb)
    train_slices, val_slices = md.get_training_slices(config, wandb)

    args = {
        'config': config,
        'wandb': wandb,
        'model': model,
        'processor': processor,
        'train_slices': train_slices,
        'val_slices': val_slices
    }

    lightning = SegSubLightning(args)

    pass
