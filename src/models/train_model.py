import os
import sys

sys.path.append(os.curdir)

import torch
import pytorch_lightning as pl

import make_lightning as ml
import src.data.make_dataset as md
import utils

torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb = utils.init_wandb()
    trainer = get_trainer(config, wandb)
    lightning = get_lightning(config, wandb)
    trainer.fit(model=lightning)
    wandb.finish()


def get_trainer(config, wandb):
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

    if wandb.config.dry:
        trainer = pl.Trainer(
            max_epochs=1,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            # accelerator='gpu',
            devices=1,
            precision='16-mixed',
            limit_train_batches=1,
            limit_val_batches=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=1,
            precision='16-mixed'
        )

    return trainer


def get_lightning(config, wandb):
    train_volumes, val_volumes = md.get_training_volumes(config, wandb)
    processor, model = ml.get_processor_model(config, wandb)

    args = {
        'config': config,
        'wandb': wandb,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }

    lightning = ml.SegSubLightning(args)

    return lightning


if __name__ == '__main__':
    main()
