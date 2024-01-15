import os
import sys

sys.path.append(os.curdir)

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice

import src.data.make_dataset as md
import utils


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        # self.save_hyperparameters(logger=False)
        self.args = args
        self.model = self.args['model']
        self.processor = self.args['processor']
        self.batch_size = self.args['wandb'].batch_size
        self.num_workers = self.args['wandb'].num_workers
        # self.val_dice = Dice(num_classes=1, threshold=0.8, average='macro')

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        return loss

    # def log_slice_mask(self, image, mask_target, mask_pred):
    #     pass

    def validation_step(self, batch, batch_idx):
        item, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        # outputs = self.processor.post_process_instance_segmentation(outputs)
        self.log('val/loss', loss, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        # self.val_dice.update(logits, y)

        return loss

    # def on_validation_epoch_end(self):
    #     self.log('val/dice', self.val_dice.compute(), on_epoch=True, sync_dist=True)
    #     self.val_dice.reset()

    # def test_step(self, batch, batch_idx):
    #     pass

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args['wandb'].lr)

        return optimizer

    def train_dataloader(self):
        args = {
            'config': self.args['config'],
            'processor': self.args['processor'],
            'set': 'train',
            'volumes': self.args['train_volumes'],
            'dim': self.args['wandb'].dim
        }

        dataset_train = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=md.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        args = {
            'config': self.args['config'],
            'processor': self.args['processor'],
            'set': 'val',
            'volumes': self.args['train_volumes'],
            'dim': self.args['wandb'].dim
        }

        dataset_val = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=md.collate_fn,
            pin_memory=True
        )

    # def test_dataloader(self):
    #     args = {'config': config, 'processor': processor, 'set': 'test', 'volumes': test_volumes, 'dim': '0,1'}
    #     dataset_test = SegSubDataset(args)
    #
    #     return DataLoader(
    #         dataset=dataset_test,
    #         batch_size=config.dataloader.batch_size,
    #         num_workers=config.dataloader.num_workers,
    #         shuffle=False,
    #         drop_last=False,
    #         collate_fn=md.collate_fn
    #     )


if __name__ == '__main__':
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
    from sklearn.model_selection import train_test_split

    config = utils.get_config()
    wandb = utils.init_wandb()

    processor = Mask2FormerImageProcessor.from_pretrained(
        wandb.config.model_id,
        do_rescale=False,
        num_labels=1
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        wandb.config.model_id,
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    test_volumes = md.get_volumes(config, set='test')
    train_volumes = md.get_volumes(config, set='train')

    train_volumes, val_volumes = train_test_split(
        train_volumes,
        test_size=wandb.config.val_size,
        random_state=wandb.config.random_state
    )

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
