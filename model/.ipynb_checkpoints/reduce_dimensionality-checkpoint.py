from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import h5py
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from multiprocessing import Process, Queue, Pool


class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, indices=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            indices (list, optional): List with integers with the index of the people included
                (to seperate people in the training and test set)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.brain_frame = pd.read_csv(csv_file)
        if indices is not None:
            l = [i for i, t in enumerate(self.brain_frame["slice_path"].to_numpy()) if int(t[t.index("volume_")+7 : t.index("_slice")]) in indices]
            print(len(l))
            self.brain_frame = self.brain_frame.iloc[l, :]
        self.brain_frame = self.brain_frame.sample(frac=1).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.brain_frame)
    
    def get_label(self, idx):
        return self.brain_frame["target"][idx]
    
    def push_item(self, q, idx):
        q.put(self.__getitem__(idx))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == slice:
            return np.array([self.__getitem__(i)[0] for i in range(len(self))[idx]]), [self.get_label(i) for i in range(len(self))[idx]]
        
        img_path = self.brain_frame["slice_path"][idx]
        img_path = img_path[img_path.index("Bra"):]
        img_path = os.path.join(self.root_dir, img_path)

        with h5py.File(img_path,'r') as hf5:
            sample = hf5["image"][:]

        if np.isnan(sample).any():
            print(f"nan i {ind}")
        sample = (sample - np.min(sample))/max(0.001, (np.max(sample) - np.min(sample)))
        if self.transform:
            sample = self.transform(sample)

        return sample, self.brain_frame["target"][idx]
    