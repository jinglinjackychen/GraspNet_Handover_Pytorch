###################################################################################################
# Get color image and depth image form the rosbag .                                               #
# parameter 'bags_dir' is the folder contains the bags source .                                   #
# parameter 'data_num' is the numbers of image you want to get from each bag .                    #
# parameter 'depth_filter' is explain if you want to filter out some pixel value of depth image . #
# This program will output color images with 3 channels and depth images with 1 channel .         #
# This program will create the 'color' and 'depth' folders in the bags_dir and save the images .  #
###################################################################################################
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import ros_numpy
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--bags_dir', type=str, default = None)
parser.add_argument('--data_num', type=int, default=1)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--depth_filter', type=bool, default=False)
args = parser.parse_args()

# Folder include bags
path = args.bags_dir + '/'

# target topic : color and depth
topic_color = '/camera/color/image_raw'
topic_depth = '/camera/aligned_depth_to_color/image_raw'

def get_image_list(bag_dir, topic):
  imgs = []
  b = rosbag.Bag(bag_dir,'r')
  i = 0
  for topic, msg, t in b.read_messages(topic):
    if i < args.data_num:
      img = ros_numpy.image.image_to_numpy(msg)
      img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
      imgs.append(img)
      i += 1
  return imgs

print('Collecting image ...')
Color = []
Depth = []
kernel = np.ones((2,2), np.uint8)

for bagname in os.listdir(path):
  for img in get_image_list(path + bagname , topic_color):
    Color.append(img)
  
  for img in get_image_list(path + bagname ,topic_depth):
    img = np.round((img/np.max(img))*255).astype('int').reshape(args.height, args.width)
    img = np.uint8(img)
    if args.depth_filter == True:
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
          if img[i,j] > 100:
            img[i,j] = 0
      img = cv2.erode(img, kernel, iterations = 1)
    img = img.reshape(img.shape[0], img.shape[1])
    Depth.append(img)


print(len(Color),' color images')
print(len(Depth),' depth images')

os.mkdir(path + 'color')
os.mkdir(path + 'depth')

for i in range(len(Color)):
  cv2.imwrite(path+'color/color_'+str(i)+'.jpg', Color[i][:,:,[2,1,0]])
  cv2.imwrite(path+'depth/depth_'+str(i)+'.jpg', Depth[i])
  i += 1

print('Done')


