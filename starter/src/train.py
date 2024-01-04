import ctypes
import pickle
import platform

system_name = platform.system()
if system_name in ["Windows", 'Linux']:
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
else:
    libgcc_s = ctypes.CDLL('libgcc_s.1.dylib')

import os
import sys
import json
import yaml
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

import random
from time import gmtime, strftime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from torch.utils.data import Dataset

from utils import get_model, get_loss, get_scheduler, get_metric
from dataset import VolumeDataset




class SegmentationModule(pl.LightningModule):
    """
    Main training module that defines model archirecture, loss function, metric,
    optimizer, constructs training and validation dataloaders.
    """
    def __init__(self, 
                config: Mapping, 
                mode: str='train',
                data_path_train: str = 'data/train/',
                data_path_val: str = 'data/test/'):
        super().__init__()
        self.validation_step_outputs = []
        self.hparams.update(config)
        self.save_hyperparameters(config)

        self.net = get_model(**self.hparams['model'])
        self.loss = get_loss(**self.hparams['loss'])
        self.metric = get_metric(**self.hparams['metric'])
        
        
        val_transform = A.Compose(
            [#A.CenterCrop(100, 300),
             A.PadIfNeeded(min_height=100 + 28, min_width=300 + 20, border_mode=0),
             A.HorizontalFlip(), 
             A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=1.0), 
             ToTensorV2()]
        )

        train_transform = A.Compose(
            [A.PadIfNeeded(min_height=100 + 28, min_width=300 + 20, border_mode=0), 
             A.RandomCrop(32, 64), 
             A.HorizontalFlip(), 
             A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=1.0),
            ToTensorV2()
            ]
        )
        
        if mode == 'train':
            val_dataset = VolumeDataset(data_path_val, val_transform)
            train_dataset = VolumeDataset(data_path_train, val_transform)
            self.train_data = train_dataset
            self.val_data = val_dataset
        else:
            val_dataset = VolumeDataset(data_path_val, val_transform)
            self.val_data = val_dataset
        

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        predict = self.forward(image)

        loss = self.loss(predict, mask)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        predict = self.forward(image)

        val_loss = self.loss(predict, mask)
        val_metric = self.metric(predict, mask)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            'val_loss': val_loss,
            'val_metric': val_metric
        }
        
        self.validation_step_outputs.append(output)
            
        #return output

    def on_validation_epoch_end(self):
        val_loss = 0
        val_metric = 0
        
        outputs = self.validation_step_outputs

        for output in outputs:
            val_loss += output['val_loss']
            val_metric += output['val_metric']

        val_loss /= len(outputs)
        val_metric /= len(outputs)
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_metric', val_metric, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            'val_loss': val_loss,
            'val_metric': val_metric
        }
        self.validation_step_outputs.clear() 

        #return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams["lr"])
        scheduler = get_scheduler(optimizer, **self.hparams['scheduler'])
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )        
    
    
def main():
    
    
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    model = SegmentationModule(config)

    time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    checkpoint_dirpath = os.path.join('checkpoints/', config['name']+'_'+time, )
    os.mkdir(checkpoint_dirpath)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dirpath, **config['model_checkpoint'])

    trainer = pl.Trainer(
                logger=model.logger,
                precision=32,
                accelerator="gpu", devices=1,
                max_epochs=config['epochs'],
                callbacks=[checkpoint_callback],
                log_every_n_steps=1
            )

    trainer.fit(model)
    
    
if __name__ == '__main__':
    main()