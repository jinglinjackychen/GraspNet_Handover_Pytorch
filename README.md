# GraspNet_Handover_Pytorch

Here is the work to use FCN model to predict a "grasp point" by Pytorch .

![Teaser](figure/demo.png)

# How to use

## I. Get image data from rosbags
You can collect the image by recording rosbag and get data from them . This work will generate two folder which include RGB images and Depth images .
```bash
$ python3 rosbag2img.py --bags_dir="your own path"
```

## II. Labeling and generate datasets
Our method of labeling is to label the grasping pose of parallel gripper . ( green line )
![Dataset](figure/dataset.png)

This command will create the lable images from json files . And create the training and testing list .
```bash
$ python3 data_preprocess.py --data_dir="your own path"
```
## II. Train
```bash
$ python3 train.py --data_dir="your own path"
```
