import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
import torch.nn.functional as F
import json
import os
from random import sample
import argparse

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--data_dir', type=str, default = None)
parser.add_argument('--split_ratio', type=float, default=0.1)
args = parser.parse_args()

# file path , contain 'color' , 'depth' , 'label' , 'json' folders
File = os.listdir(args.data_dir+'/json')
File.sort()

# create data name list
name = os.listdir(args.data_dir+'/color')
name_list = []
for num in name:
  name_list.append(num.split('_')[1].split('.')[0])

if os.path.isdir(args.data_dir + '/label') == False:
    os.mkdir(args.data_dir + '/label')
  
# draw label
for name in File:
  label = np.zeros((256,256,3))
  with open(args.data_dir+'/json'+'/'+name,"r") as f:
    data = json.load(f)
    for i in range(len(data['shapes'])):
      coord = data['shapes'][i]['points']
      if data['shapes'][i]['label'] == 'good':
        cv2.line(label, (int(coord[0][0]), int(coord[0][1])), (int(coord[1][0]), int(coord[1][1])), (0,255,0),2)

  cv2.imwrite(args.data_dir+'/label/label_'+name.split('.')[0].split('_')[1]+'.jpg', label[:,:,[2,1,0]])

# flip 3 times
temp = [0, 1, -1]
for idx in name_list:
  color = cv2.imread(args.data_dir+'/color/color_'+idx+'.jpg')
  depth = cv2.imread(args.data_dir+'/depth/depth_'+idx+'.jpg')
  label = cv2.imread(args.data_dir+'/label/label_'+idx+'.jpg')
  for n in range(-1,2):
    color_ = cv2.flip(color,n)
    depth_ = cv2.flip(depth,n)
    label_ = cv2.flip(label,n)
    cv2.imwrite(args.data_dir+'/color/color_'+idx+'_'+str(n+1)+'.jpg',color_)
    cv2.imwrite(args.data_dir+'/depth/depth_'+idx+'_'+str(n+1)+'.jpg',depth_)
    cv2.imwrite(args.data_dir+'/label/label_'+idx+'_'+str(n+1)+'.jpg',label_)

# Create training & testing list
Path = args.data_dir +'/color'
File = os.listdir(Path)
File.sort()

data_list = []
for name in File:
  data_list.append(name.split('color')[1])

test = sample(data_list, int(len(data_list)*args.split_ratio))

train = list(set(data_list).difference(set(test)))
f = open(args.data_dir+'/test.txt', "a")
for idx in test:
  f.write(idx+'\n')
f.close()

f = open(args.data_dir+'/train.txt', "a")
for idx in train:
  f.write(idx+'\n')
f.close()