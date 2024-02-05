import os
import random
import time

import numpy as np

import src.utils as utils


def main():
    config = utils.get_config()
    submission_name = '3xg8r6lz_2snz8a1d'

    test_vol_file = 'test_vol_41.npy'
    test_vol_path = os.path.join(config['path']['data']['raw']['test'], test_vol_file)
    test_vol = np.load(test_vol_path, allow_pickle=True)

    sub_vol_file = test_vol_file.replace('test', 'sub')
    sub_vol_path = os.path.join(config['path']['submissions']['root'], submission_name, sub_vol_file)
    sub_vol = np.load(sub_vol_path, allow_pickle=True)

    indexes = random.choices(range(len(test_vol)), k=10)

    for i in indexes:
        utils.plot_slice(np.transpose(test_vol[i]), title=f'Test slice {i}')
        utils.plot_slice(np.transpose(sub_vol[i]), title=f'Sub slice {i}')
        time.sleep(0.5)


if __name__ == '__main__':
    main()
