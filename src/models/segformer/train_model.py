import os
import warnings

import pytorch_lightning as pl
import torch
import wandb

import src.data.make_dataset as md
import src.models.segformer.make_lightning as segformer_ml
from src import utils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('segformer.yml')
    trainer = get_trainer(config)
    lightning = get_lightning(config, wandb_config, checkpoint='swift-sponge-567-k13mlcpr.ckpt')
    trainer.fit(model=lightning)
    wandb.finish()


def get_trainer(config):
    os.makedirs(config['path']['models']['root'], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/dice',
        mode='max',
        dirpath=config['path']['models']['root'],
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    if wandb.config.dry:
        trainer = pl.Trainer(
            max_epochs=3,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            devices=1,
            precision='16-mixed',
            limit_train_batches=5,
            limit_val_batches=5
        )
    else:
        trainer = pl.Trainer(
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            precision='16-mixed'
        )

    return trainer


def get_lightning(config, wandb_config, checkpoint=None):
    train_volumes, val_volumes = md.get_training_volumes(config, wandb_config)
    processor = utils.get_processor(config, wandb_config)
    model = segformer_ml.get_model(wandb_config)

    args = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes
    }

    if checkpoint is None:
        lightning = segformer_ml.SegSubLightning(args)
    else:
        path_checkpoint = os.path.join(config['path']['models']['root'], checkpoint)
        lightning = segformer_ml.SegSubLightning.load_from_checkpoint(path_checkpoint, args=args)

    return lightning


if __name__ == '__main__':
    main()
