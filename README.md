# M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection
## Introduction
This is the official PyTorch implementation of M2F2-Net: Multi-Modal Feature Fusion for Unstructured Off-Road Freespace Detection

![result_1](https://user-images.githubusercontent.com/70512651/217461427-2d194fb0-b3b9-489b-8569-69cfaa6d9eb7.png)
![result_2](https://user-images.githubusercontent.com/70512651/217461448-43530fa4-5d45-4df3-8daf-1c105d79337b.png)
![result_3](https://user-images.githubusercontent.com/70512651/217461463-90ac9048-7468-49a6-a1d1-c902b0cdaeb9.png)


Freespace detection is an important part of autonomous driving technology. Compared with structured on-road scenes, unstructured off-road scenes face more challenges.  Multi-modal fusion method is a viable solution to these challenges. But existing fusion methods do not fully utilize the multi-modal features. In this paper, we propose an effective multi-modal network named M2F2-Net for freespace detection in unstructured off-road scenes. We propose a multi-modal feature fusion strategy named Multi-modal Cross Fusion (MCF). MCF module is simple but effective in fusing the features of RGB images and surface normal maps. Meanwhile, a multi-modal segmentation decoder module is designed to decouple the segmentation of two modalities, and it further helps the features of both modalities to be fully utilized. In order to solve the problem that the road edge is difficult to extract in the unstructured scenes, we also propose an edge segmentation decoder module.

![image](https://user-images.githubusercontent.com/70512651/216315067-867ec5af-e27a-492f-98f0-2523725f51ad.png)

## Requirements
- python 3.8
- pytorch 1.9
- pip install mmcv, mmsegmentation
- pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

## Pretrained Models
The pretrained models of our OFF-Net trained on ORFD dataset can be download from [here](https://pan.baidu.com/s/1_Mb8jx8KoR9n11M_lfm4cg)(Extract code：djbv)

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

## Usage
### Image_Demo
```
python demo.py
```

### Video_Demo
```
python demo_video.py
```

### Training
```
python train.py --data_root ORFD_Dataset_ICRA2022/ --exp_name test --start_epoch 0 --gpu_ids 0
```

### Test
```
python test.py --data_root ORFD_Dataset_ICRA2022/ --exp_name test --gpu_ids 0
```

## Acknowledgement
Our code is inspired by [SegFormer](https://github.com/NVlabs/SegFormer), [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection) and [Pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)
