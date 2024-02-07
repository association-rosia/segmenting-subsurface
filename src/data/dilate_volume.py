import numpy as np

import make_dataset as md
from src import utils


def main():
    config = utils.get_config()
    train_volumes = md.get_volumes(config, set='train')

    for volume_path in train_volumes:
        volume = np.load(volume_path, allow_pickle=True)
        label_path = volume_path.replace('seismic', 'horizon_labels')
        label = np.load(label_path, allow_pickle=True)

        if volume.shape[0] == 300:  # to divide by 30 the number of slices
            indexes = [i for i in range(0, len(volume), 30)]
            volume = volume[indexes]
            np.save(volume_path, volume, allow_pickle=True)
            print(f'Resize {volume_path}')

        if label.shape[0] == 300:  # to divide by 30 the number of slices
            indexes = [i for i in range(0, len(label), 30)]
            label = label[indexes]
            np.save(label_path, label, allow_pickle=True)
            print(f'Resize {label_path}')


if __name__ == '__main__':
    main()
