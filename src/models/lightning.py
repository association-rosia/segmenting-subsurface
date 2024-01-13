import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice

import src.data.make_dataset as md
import utils
from src.data.make_dataset import SegSubDataset


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.args = args
        self.metrics = Dice(num_classes=1, threshold=0.8, average='macro')

        self.save_hyperparameters(logger=False)

    def forward(self, inputs):
        outputs = self.model(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        return outputs

    def training_step(self, batch):
        outputs = self.forward(batch)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    # def log_slice_mask(self, image, mask_target, mask_pred):
    #     pass

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

    # def test_step(self, batch, batch_idx):
    #     pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args['config']['training']['lr'])

        return optimizer

    def train_dataloader(self):
        args = {
            'config': self.args['config'],
            'processor': self.args['processor'],
            'set': 'train',
            'volumes': self.args['train_volumes'],
            'dim': self.args['dim']
        }

        dataset_train = SegSubDataset(args)

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.args['config']['dataloader']['batch_size'],
            num_workers=self.args['config']['dataloader']['num_workers'],
            shuffle=True,
            drop_last=True,
            collate_fn=md.collate_fn
        )

    def val_dataloader(self):
        args = {
            'config': self.args['config'],
            'processor': self.args['processor'],
            'set': 'val',
            'volumes': self.args['train_volumes'],
            'dim': self.args['dim']
        }

        dataset_val = SegSubDataset(args)

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.args['config']['dataloader']['batch_size'],
            num_workers=self.args['config']['dataloader']['num_workers'],
            shuffle=False,
            drop_last=True,
            collate_fn=md.collate_fn
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
    model_id = 'facebook/mask2former-swin-large-coco-instance'
    processor = Mask2FormerImageProcessor.from_pretrained(model_id, num_labels=1)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, num_labels=1, ignore_mismatched_sizes=True)

    test_volumes = md.get_volumes(config, set='test')
    train_volumes = md.get_volumes(config, set='train')
    train_volumes, val_volumes = train_test_split(train_volumes,
                                                  test_size=config['training']['val_size'],
                                                  random_state=config['random_state'])

    args = {
        'config': config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes,
        'dim': '0,1'
    }

    lightning = SegSubLightning(args)

    pass
