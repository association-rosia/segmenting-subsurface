import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice
from transformers import Mask2FormerForUniversalSegmentation

import src.data.make_dataset as md
import utils
from src.data.make_dataset import SegSubDataset


class SegSubLightning(pl.LightningModule):
    """
    Lightning Module for the Segmenting-Subsurface project.
    """

    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.args = args

        self.save_hyperparameters(logger=False)

        # m2f_config = Mask2FormerConfig()
        self.model = Mask2FormerForUniversalSegmentation()

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

        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.metrics.reset()

        return metrics

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer

    def train_dataloader(self):
        dataset_train = SegSubDataset()

        return DataLoader(
            dataset=dataset_train,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=md.collate_fn
        )

    def val_dataloader(self):
        dataset_val = SegSubDataset()

        return DataLoader(
            dataset=dataset_val,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=md.collate_fn
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
            collate_fn=md.collate_fn
        )


if __name__ == '__main__':
    config = utils.get_config()
