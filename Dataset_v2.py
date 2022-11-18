import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from random import shuffle
import pandas as pd


class PixWiseDataset():
    def __init__(self, csvfile, map_size=14,
                 smoothing=True, transform=None):
        self.data = pd.read_csv(csvfile)
        self.transform = transform
        self.map_size = map_size
        self.label_weight = 0.99 if smoothing else 1.0
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        if torch.is_tensor(ind):
            ind = ind.tolist()

        img_name = self.data.iloc[ind]['name']
        img = Image.open(img_name)
        label = self.data.iloc[ind]['label']
        if label == 0:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

        if self.transform:
            img = self.transform(img)
        label = np.array([label])
        sample = {'image': img, 'label': label, 'mask':mask}
        return sample
        
