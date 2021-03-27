import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torch.utils import model_zoo
import random
import sys
import copy
import math
import pandas as pd

import torchvision.transforms as transforms

import cv2
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import numpy as np
import time
import os


image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

class parallel_jaw_based_grasping_dataset(Dataset):
    name = []
    def __init__(self, data_dir, scale=1/(6.4), mode='train'):
        self.data_dir = data_dir
        self.scale = 1/(6.4)
        self.mode = mode
        if self.mode == 'train':
            f = open(self.data_dir+"/train.txt", "r")
        else:
            f = open(self.data_dir+"/test.txt", "r")
        for i, line in enumerate(f):
              self.name.append(line.replace("\n", ""))
    def __len__(self):
        return len(self.name)
    def __getitem__(self, idx):
        idx_name = self.name[idx]
        color_img = cv2.imread(self.data_dir+"/color/color"+idx_name)
        color_img = color_img[:,:,[2,1,0]]
        depth_img = cv2.imread(self.data_dir+"/depth/depth"+idx_name, 0)

        label_img = cv2.imread(self.data_dir+"/label/label"+idx_name, cv2.IMREAD_GRAYSCALE)
        # uint8 -> float
        color = (color_img/255.).astype(float)
        # BGR -> RGB and normalize
        color_rgb = np.zeros(color.shape)
        for i in range(3):
            color_rgb[:, :, i] = (color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]
        depth = (depth_img/1000.).astype(float) # to meters
        # SR300 depth range
        depth = np.clip(depth, 0.0, 1.2)
        # Duplicate channel and normalize
        depth_3c = np.zeros(color.shape)
        for i in range(3):
            depth_3c[:, :, i] = (depth[:, :]-image_net_mean[i])/image_net_std[i]
        # Unlabeled -> 2; unsuctionable -> 0; suctionable -> 1
        label = np.round(label_img/255.*2.).astype(float)
        # Already 40*40
        label = cv2.resize(label, (int(32), int(32)))
        transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        color_tensor = transform(color_rgb).float()
        depth_tensor = transform(depth_3c).float()
        label_tensor = transform(label).float()
        sample = {"color": color_tensor, "depth": depth_tensor, "label": label_tensor}
        return sample