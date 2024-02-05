import os
import time

import numpy as np

import src.utils as utils


def main():
    config = utils.get_config()
    submission_name = '3xg8r6lz_2snz8a1d'
    volume_file = 'sub_vol_41.npy'
    volume_path = os.path.join(config['path']['submissions']['root'], submission_name, volume_file)
    volume = np.load(volume_path, allow_pickle=True)

    print(len(volume))
    for i in range(len(volume)):
        utils.plot_slice(np.transpose(volume[i]))
        time.sleep(0.5)

    print()


if __name__ == '__main__':
    main()
