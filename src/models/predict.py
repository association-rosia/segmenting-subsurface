# import src.models.segformer.make_lightning as segformer_ml
# import src.models.mask2former.make_lightning as mask2former_ml
# import src.models.segment_anything.make_lightning as segment_anything_ml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SegSubPredictDataset(Dataset):
    def __init__(self, volumes):
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        item = np.load(self.volumes[idx], allow_pickle=True)
        item = torch.from_numpy(item)
        item = item.to(dtype=torch.float32)

        return item


def get_test_volumes():



def main():
    segformer_id = None
    mask2former_id = None
    segment_anything_id = None

    volumes = get_test_volumes()

    predict_dataset = SegSubPredictDataset(volumes)
    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=1,
        shuffle=False
    )

    for i, volume in enumerate(predict_dataloader):
        pass
        # if segformer_id:
        # load segformer processor & model
        # for slice in slice (use SegSubDataset)
        #   segformer_output = process volume + predict binary mask

        # if mask2former_id:
        # load mask2former processor & model
        # for slice in slice (use SegSubDataset)
        #   mask2former_output = process volume & segformer_output + predict instance mask

        # if mask2former_id and segment_anything_id:
        # load segment_anything processor & model
        # for slice in slice (use SegSubDataset)
        #   segment_anything_output = process volume & mask2former_output + predict instance mask

        # if not mask2former_id and segment_anything_id:
        # load segment_anything processor & model
        # for slice in slice (use SegSubDataset)
        #   segment_anything_output = process volume & segformer_output + predict instance mask


if __name__ == '__main__':
    main()
