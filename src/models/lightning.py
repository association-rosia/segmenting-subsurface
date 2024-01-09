from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Dice
from torch.utils.data import DataLoader
from src.data.make_dataset import SegSubDataset
from utils import get_config

config = get_config()


class SegSubLightning(pl.LightningModule):
    """
    Lightning Module for the Segmenting-Subsurface project.
    """

    def __init__(
            self,
    ):
        super(SegSubLightning, self).__init__()
        self.save_hyperparameters(logger=False)

        m2f_config = Mask2FormerConfig(
            # feature_size=300,
            # mask_feature_size=300,
            # hidden_dim=300
        )
        self.model = Mask2FormerForUniversalSegmentation(m2f_config)

        # init metrics for evaluation
        self.metrics = Dice(num_classes=1, threshold=0.8, average='macro')

    def forward(self, inputs):
        outputs = self.model(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,  # we need the intermediate hidden states
            return_dict=True,
        )
        
        return outputs

    def training_step(self, batch):
        outputs = self.forward(batch)
        
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def log_slice_mask(self, image, mask_target, mask_pred):
        pass

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        loss = outputs['loss']
        self.log('val/loss', loss, on_step=True, on_epoch=True)
        # self.metrics.update(outputs, )

        # if batch_idx == self.log_image_idx // self.train_batch_size:
        #     batch_image_idx = self.log_image_idx % self.train_batch_size
            # self.log_slice_mask(aerial[batch_image_idx], labels[batch_image_idx], outputs[batch_image_idx])

        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute metrics
        metrics = self.metrics.compute()

        # Log metrics
        self.log_dict(metrics, on_epoch=True)

        # Reset metrics
        self.metrics.reset()

        return metrics

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        # scheduler = ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     verbose=True
        # )

        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'}}
        return optimizer

    def train_dataloader(self):
        # Initialize training dataset and data loader
        dataset_train = SegSubDataset()

        return DataLoader(
            dataset=dataset_train,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # Initialize validation dataset and data loader
        dataset_val = SegSubDataset()

        return DataLoader(
            dataset=dataset_val,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        # Initialize test dataset and data loader
        dataset_test = SegSubDataset()

        return DataLoader(
            dataset=dataset_test,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False,
            drop_last=False,
        )

