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
import math

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
from Dataloader import *
import argparse


parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--data_dir', type=str, default = None)
parser.add_argument('--epoch', type=int, default = 50)
parser.add_argument('--save_every', type=int, default = 5)
parser.add_argument('--batch_size', type=int, default = 10)
args = parser.parse_args()

if os.path.isdir(args.data_dir + '/weight') == False:
    os.mkdir(args.data_dir + '/weight')


dataset = parallel_jaw_based_grasping_dataset(args.data_dir)
dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)

class_weight = torch.ones(2)
# class_weight[2] = 0
net = GraspNet(2)
net = net.cuda()
criterion = nn.CrossEntropyLoss(class_weight).cuda()
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.1)



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

        loss = criterion(predict.view(len(sampled_batched['color']), 2,32*32), label.view(len(sampled_batched['color']), 32*32))
        loss.backward()
        loss_sum += loss.detach().cpu().numpy()
        optimizer.step()
    scheduler.step()
    loss_l.append(loss_sum/len(dataloader))
    if (epoch+1)%args.save_every==0:
        torch.save(net.state_dict(), args.data_dir + '/weight/grapnet_{}_{}.pth' .format(epoch+1, loss_l[-1]))

    print("Epoch: {}| Loss: {}| Time elasped: {}".format(epoch+1, loss_l[-1], time.time()-ts))