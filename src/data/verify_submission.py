import os
import random
import time

import numpy as np

import src.utils as utils


def main():
    config = utils.get_config()
    submission_name = 'prime-fog-209-xzs93mfw'
    sub_vol_pathes = os.path.join(config['path']['submissions']['root'], submission_name)
    sub_vol_files = os.listdir(sub_vol_pathes)

    for sub_vol_file in sub_vol_files:
        sub_vol_path = os.path.join(sub_vol_pathes, sub_vol_file)
        sub_vol = np.load(sub_vol_path, allow_pickle=True)
        index = random.choices(range(len(sub_vol)), k=1)
        utils.plot_slice(np.transpose(sub_vol[index]), title=f'Sub slice {index}')
        time.sleep(0.5)


if __name__ == '__main__':
    main()
