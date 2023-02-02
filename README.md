# M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection
## Introduction
This is the official PyTorch implementation of M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection
![image](https://user-images.githubusercontent.com/70512651/216315067-867ec5af-e27a-492f-98f0-2523725f51ad.png)
## Datasets
The ORFD dataset we used can be found [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection). Extract and organize as follows:
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
