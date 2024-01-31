# import src.models.segformer.make_lightning as segformer_ml
# import src.models.mask2former.make_lightning as mask2former_ml
# import src.models.segment_anything.make_lightning as segment_anything_ml

from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.make_dataset as md
import src.utils as utils


def main():
    segformer_id = '4u9c0boz'
    mask2former_id = segformer_id
    segment_anything_id = None

    config = utils.get_config()
    run = get_run_from_model_id(mask2former_id, segment_anything_id)
    model, processor = load_model_processor(config, run, mask2former_id, segment_anything_id)
    test_volumes = md.get_volumes(config, set='test')

    args = {
        'config': config,
        'wandb_config': run.config,
        'processor': processor,
        'volumes': test_volumes,
        'set': 'test'
    }

    test_dataset = md.SegSubDataset(args)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=300,
        shuffle=False
    )

    for item, inputs in tqdm(test_dataloader):
        pass


def load_model_processor(config, run, mask2former_id, segment_anything_id):
    lightning = None

    if mask2former_id and segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both defined: {mask2former_id} - {segment_anything_id}")
    elif not mask2former_id and not segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both null")
    elif mask2former_id:
        lightning = utils.load_segformer_model(config, run)
    else:
        lightning = utils.load_segment_anything(config, run)

    model = lightning.model
    processor = lightning.processor

    return model, processor


def get_run_from_model_id(mask2former_id, segment_anything_id):
    run = None
    
    if mask2former_id and segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both defined: {mask2former_id} - {segment_anything_id}")
    elif not mask2former_id and not segment_anything_id:
        ValueError(f"Mask2former and SAM can't be both null")
    elif mask2former_id:
        run = utils.get_run(mask2former_id)
    else:
        run = utils.get_run(segment_anything_id)

    run.config['dilation'] = 1

    return run


if __name__ == '__main__':
    main()
