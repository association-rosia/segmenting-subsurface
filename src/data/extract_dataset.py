import os
import sys

sys.path.append(os.curdir)

import utils
import zipfile
from tqdm import tqdm


def main():
    config = utils.get_config()
    create_folders(config)
    extract_files(config)


def create_folders(config):
    os.makedirs(config['path']['data']['raw']['train'], exist_ok=True)
    os.makedirs(config['path']['data']['raw']['test'], exist_ok=True)


def get_extract_path(config, file):
    set = file.split('-data')[0].split('layer-')[-1]
    path = config['path']['data']['raw'][set]

    return path


def extract_files(config):
    path = config['path']['data']['raw']['root']
    files = [f for f in os.listdir(path) if f.split('.')[-1] == 'zip']

    for file in tqdm(files):
        file_path = os.path.join(path, file)
        with zipfile.ZipFile(file_path, 'r') as f:
            extract_path = get_extract_path(config, file)
            f.extractall(extract_path)
            os.remove(file_path)


if __name__ == '__main__':
    main()
