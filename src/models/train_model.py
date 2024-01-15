import os
import sys

sys.path.append(os.curdir)

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

import make_lightning as ml
import src.data.make_dataset as md
import utils

torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb = utils.init_wandb()

    os.makedirs(config['path']['models']['root'], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=config['path']['models']['root'],
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=wandb.config.max_epochs,
        logger=pl.loggers.WandbLogger(),
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=0,
        # strategy='ddp',
        # precision='16-mixed'
    )

    processor = Mask2FormerImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        do_rescale=False,
        num_labels=1
    )

    # TODO: set num_channels to 1
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    # test_volumes = md.get_volumes(config, set='test')
    train_volumes = md.get_volumes(config, set='train')
    train_volumes, val_volumes = train_test_split(
        train_volumes,
        test_size=wandb.config.val_size,
        random_state=wandb.config.random_state
    )

    args = {
        'config': config,
        'wandb': wandb.config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }

    lightning = ml.SegSubLightning(args)

    trainer.fit(model=lightning)

    wandb.finish()


if __name__ == '__main__':
    main()
