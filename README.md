# M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection
## Introduction
This is the official PyTorch implementation of M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection

Freespace detection is an important part of autonomous driving technology. Compared with structured on-road scenes, unstructured off-road scenes face more challenges.  Multi-modal fusion method is a viable solution to these challenges. But existing fusion methods do not fully utilize the multi-modal features. In this paper, we propose an effective multi-modal network named M2F2-Net for freespace detection in unstructured off-road scenes. We propose a multi-modal feature fusion strategy named Multi-modal Cross Fusion (MCF). MCF module is simple but effective in fusing the features of RGB images and surface normal maps. Meanwhile, a multi-modal segmentation decoder module is designed to decouple the segmentation of two modalities, and it further helps the features of both modalities to be fully utilized. In order to solve the problem that the road edge is difficult to extract in the unstructured scenes, we also propose an edge segmentation decoder module.

![image](https://user-images.githubusercontent.com/70512651/216315067-867ec5af-e27a-492f-98f0-2523725f51ad.png)

## Requirements
- python 3.8
- pytorch 1.9
- pip install mmcv, mmsegmentation
- pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

## Datasets
The ORFD dataset we used can be found at [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection). Extract and organize as follows:
```
|-- datasets
 |  |-- ORFD
 |  |  |-- training
 |  |  |  |-- sequence   |-- calib
 |  |  |                 |-- sparse_depth
 |  |  |                 |-- dense_depth
 |  |  |                 |-- lidar_data
 |  |  |                 |-- image_data
 |  |  |                 |-- gt_image
 ......
 |  |  |-- validation
 ......
 |  |  |-- testing
 ......
```
