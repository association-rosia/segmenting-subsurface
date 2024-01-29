# import src.models.segformer.make_lightning as segformer_ml
# import src.models.mask2former.make_lightning as mask2former_ml
# import src.models.segment_anything.make_lightning as segment_anything_ml

from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.make_dataset as md
import src.utils as utils


def main():
    segformer_id = None
    mask2former_id = None
    segment_anything_id = None

    config = utils.get_config()
    wandb_config = {}  # based on segformer_id
    test_volumes = md.get_volumes(config, set='test')
    segformer_processor = utils.get_processor(config, wandb_config)

    args = {
        'config': config,
        'wandb_config': wandb_config,  # based on segformer_id
        'processor': segformer_processor,  # based on segformer_id
        'volumes': test_volumes,
    }

    test_dataset = md.SegSubDataset(args)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=wandb_config['batch_size'],
        shuffle=False
    )

    for item, inputs in tqdm(test_dataloader):
        pass
        # if segformer_id:
        # load segformer processor & model
        # for slice in slice (use SegSubDataset)
        #   segformer_output = process volume + predicted binary mask

        # if mask2former_id:
        # load mask2former processor & model
        # for slice in slice (use SegSubDataset)
        #   mask2former_output = process volume & segformer_output + predicted instance mask

        # if mask2former_id and segment_anything_id:
        # load segment_anything processor & model
        # for slice in slice (use SegSubDataset)
        #   segment_anything_output = process volume & mask2former_output + predicted instance mask

        # if not mask2former_id and segment_anything_id:
        # load segment_anything processor & model
        # for slice in slice (use SegSubDataset)
        #   segment_anything_output = process volume & segformer_output + predicted instance mask


if __name__ == '__main__':
    main()
