import os

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

import src.data.make_dataset as md
import utils
from make_lightning import SegSubLightning


def main():
    config = utils.get_config()
    wandb = utils.init_wandb()

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

    trainer = pl.Trainer(
        max_epochs=1,
        logger=pl.loggers.WandbLogger(),
        callbacks=[checkpoint_callback],
        accelerator=utils.get_device(),
        # devices=4,
        # strategy='ddp',
        precision='16-mixed'
    )

    processor = Mask2FormerImageProcessor.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        num_labels=1
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=wandb.config.model_id,
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    # test_volumes = md.get_volumes(config, set='test')
    train_volumes = md.get_volumes(config, set='train')
    train_volumes, val_volumes = train_test_split(
        train_volumes,
        test_size=config['training']['val_size'],
        random_state=config['random_state']
    )

    args = {
        'config': config,
        'model': model,
        'processor': processor,
        'train_volumes': train_volumes,
        'val_volumes': val_volumes,
        'dim': wandb.config.dim
    }

    lightning = SegSubLightning(args)

    trainer.fit(model=lightning)

    wandb.finish()


if __name__ == '__main__':
    main()
