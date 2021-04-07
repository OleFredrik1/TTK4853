import os
import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from skimage import transform


class UnprocessedDataset(Dataset):
    def __init__(self, root_dir):
        self.files = os.listdir(root_dir)
        self.files = [file for file in self.files if file[-4:] != ".csv"]
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.files)
    
    def __get_file_path(self, idx, postfix):
        return self.root_dir + "/" + self.files[idx] + "/" + self.files[idx] + "_" + postfix + ".nii"
    
    def __getitem__(self, index):
        if type(index) == tuple:
            idx, s = index
        else:
            idx, s = index, slice(None, None)
        images = [
            nib.load(self.__get_file_path(idx, postfix)) 
            for postfix in ["flair", "t1", "t2", "t1ce"]
        ]
        images = np.array([np.asanyarray(image.dataobj) for image in images])
        images = np.swapaxes(images, 0, 3)
        images = images[s]
        return images
    
    
def transform_datapoint(datapoint, mean, std):
    datapoint = transform.resize(datapoint, (224, 224, 3))
    datapoint = (datapoint - mean) / std
    datapoint = np.moveaxis(datapoint, 2, 0)
    return torch.tensor(datapoint[None, :, :, :])