import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.volumes_list = [x for x in os.listdir(data_dir) if x not in '.ipynb_checkpoints']
        self.transform = transform
        self.seismic_list = list()
        self.labels_list = list()
        
        for volume in tqdm(self.volumes_list):
            for seismic in glob(f"{data_dir}/{volume}/seismic_*.npy"):
                label = seismic.replace('seismic', 'horizon_labels')
                self.seismic_list.append(np.load(seismic, mmap_mode='r', allow_pickle=True))
                self.labels_list.append(np.load(label, mmap_mode='r', allow_pickle=True))
            
        self.volume_shape = self.seismic_list[0].shape
     
    def _convert_to_tensor(self, image, mask):
        image_dtype = torch.float32

        image = torch.from_numpy(np.array(image, dtype='float32')).type(image_dtype)
        mask = torch.from_numpy(np.array(mask, dtype='uint8')).type(torch.long)

        return image, mask
    
    def __len__(self):
        return len(self.seismic_list) * self.volume_shape[0]
    

    def __getitem__(self, index):
        
        volume_num = index // self.volume_shape[0]
        index_num = index % self.volume_shape[0]
        
        image = np.array(self.seismic_list[volume_num][index_num].T, dtype='float64')
        mask = np.array(self.labels_list[volume_num][index_num].T, dtype='long')
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return self._convert_to_tensor(image, mask)

