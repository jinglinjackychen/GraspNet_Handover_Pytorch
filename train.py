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
from model import *
import argparse

image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--data_dir', type=str, default = None)
parser.add_argument('--epoch', type=int, default = 50)
parser.add_argument('--save_every', type=int, default = 5)
parser.add_argument('--batch_size', type=int, default = 10)
args = parser.parse_args()

if os.path.isdir(args.data_dir + '/weight') == False:
    os.mkdir(args.data_dir + '/weight')

class parallel_jaw_based_grasping_dataset(Dataset):
    name = []
    def __init__(self, data_dir, scale=1/(6.4)):
        self.data_dir = data_dir
        self.scale = 1/(6.4)
        f = open(self.data_dir+"/train.txt", "r")
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

dataset = parallel_jaw_based_grasping_dataset(args.data_dir)

class_weight = torch.ones(3)
class_weight[2] = 0
net = GraspNet(3)
net = net.cuda()
criterion = nn.CrossEntropyLoss(class_weight).cuda()
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.1)

dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)


loss_l = []
for epoch in range(args.epoch):
    loss_sum = 0.0
    ts = time.time()
    for i_batch, sampled_batched in enumerate(dataloader):
        print("\r[{:03.2f} %]".format(i_batch/float(len(dataloader))*100.0), end="\r")
        optimizer.zero_grad()
        color = sampled_batched['color'].cuda()
        depth = sampled_batched['depth'].cuda()
        label = sampled_batched['label'].cuda().long()
        predict = net(color, depth)

        loss = criterion(predict.view(len(sampled_batched['color']), 3,32*32), label.view(len(sampled_batched['color']), 32*32))
        loss.backward()
        loss_sum += loss.detach().cpu().numpy()
        optimizer.step()
    scheduler.step()
    loss_l.append(loss_sum/len(dataloader))
    if (epoch+1)%args.save_every==0:
        torch.save(net.state_dict(), args.data_dir + '/weight/grapnet_{}_{}.pth' .format(epoch+1, loss_l[-1]))

    print("Epoch: {}| Loss: {}| Time elasped: {}".format(epoch+1, loss_l[-1], time.time()-ts))