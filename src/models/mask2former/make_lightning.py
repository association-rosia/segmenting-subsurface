import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics as tm
from transformers import Mask2FormerForUniversalSegmentation

import src.data.make_dataset as md
from src import utils
import wandb


class SegSubLightning(pl.LightningModule):
    def __init__(self, args):
        super(SegSubLightning, self).__init__()
        self.config = args['config']
        self.wandb_config = args['wandb_config']
        self.model = args['model']
        self.processor = args['processor']
        self.train_volumes = args['train_volumes']
        self.val_volumes = args['val_volumes']
        
        self.metrics = self.configure_metrics()

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch):
        _, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('val/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.metrics.update(outputs['masks_queries_logits'], inputs['mask_labels'])
        
        if batch_idx == 0:
            self.log_image(inputs, outputs)

        return loss
    
    def log_image(self, inputs, outputs):
        pixel_values = inputs['pixel_values'][0][0]
        outputs = self.processor.post_process_instance_segmentation(outputs)[0].numpy(force=True)
        ground_truth = inputs['mask_labels'][0].numpy(force=True)

        wandb.log(
            {'val/prediction': wandb.Image(pixel_values, masks={
                'predictions': {
                    'mask_data': outputs
                },
                'ground_truth': {
                    'mask_data': ground_truth
                }
            })})
        
    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.wandb_config['lr'])

        return optimizer
    
    def configure_metrics(self):
        num_labels = self.wandb_config['num_labels']

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
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.wandb_config['num_workers'],
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        args = {
            'config': self.config,
            'wandb_config': self.wandb_config,
            'processor': self.processor,
            'volumes': self.val_volumes
        }

        dataset_val = md.SegSubDataset(args)

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.wandb_config['num_workers'],
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )


def get_model(wandb_config):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=wandb_config['model_id'],
        num_labels=wandb_config['num_labels'],
        ignore_mismatched_sizes=True
    )

    return model



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    config = utils.get_config()
    wandb_config = utils.init_wandb(yml_file='mask2former.yml')

    processor = utils.get_processor(config, wandb_config)

    model = get_model(wandb_config)

    test_volumes = md.get_volumes(config, set='test')
    train_volumes = md.get_volumes(config, set='train')

    train_volumes, val_volumes = train_test_split(
        train_volumes,
        test_size=wandb.config.val_size,
        random_state=wandb.config.random_state
    )

    args = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }

    lightning = SegSubLightning(args)
