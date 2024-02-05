import numpy as np
from tqdm import tqdm

import make_dataset as md
from src import utils


def main():
    config = utils.get_config()
    train_volumes = md.get_volumes(config, set='train')

    for volume_path in tqdm(train_volumes):
        volume = np.load(volume_path, allow_pickle=True)
        indexes = [i for i in range(0, len(volume), 30)]
        volume = volume[indexes]
        np.save(volume_path, volume, allow_pickle=True)


if __name__ == '__main__':
    main()
