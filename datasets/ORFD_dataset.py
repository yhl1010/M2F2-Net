import os.path
import random
import torchvision.transforms as transforms
import torch
from torch.utils import data
import cv2
import numpy as np
import glob
from datasets.target_generation import generate_edge


class orfddataset(data.Dataset):
    """dataloader for ORFD dataset"""
    def __init__(self, root, mode='train', use_size=[1280, 704], transform=True):
        self.root = root # path for the dataset
        self.use_size = np.asarray(use_size)
        self.mode = mode
        self.transform = transform

        if self.mode == "train":
            self.image_list = glob.glob(os.path.join(self.root, 'training', '*','image_data', '*.png')) # shape: 1280*720
        elif self.mode == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', '*','image_data', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', '*','image_data', '*.png')))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.imread(os.path.join(useDir, 'image_data', name))
        sn_image = cv2.imread(os.path.join(useDir, 'surface_normal', name))

        oriHeight, oriWidth, _ = rgb_image.shape

        # label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label_img_name = name.split('.')[0] + "_fillcolor.png"
        label_dir = os.path.join(useDir, 'gt_image', label_img_name)
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        label = np.where(label > 200, 1, 0)

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, (self.use_size[0], self.use_size[1]))
        label = cv2.resize(label, (int(self.use_size[0]/4), int(self.use_size[1]/4)), interpolation=cv2.INTER_NEAREST)
        sn_image = cv2.resize(sn_image, (self.use_size[0], self.use_size[1]))

        label_edge = generate_edge(label)

        if self.transform:
            rgb_image = (rgb_image.astype(np.float32) / 255 - 0.5) / 0.5
            rgb_image = transforms.ToTensor()(rgb_image)

            sn_image = (sn_image.astype(np.float32) / 255 - 0.5) / 0.5
            sn_image = transforms.ToTensor()(sn_image)

            label = torch.from_numpy(label)
            label = label.type(torch.LongTensor)

            label_edge = torch.from_numpy(label_edge)
            label_edge = label_edge.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'sn_image': sn_image, 'label': label, 'label_edge':label_edge,
                'path': label_dir, 'oriSize': (oriWidth, oriHeight)}


if __name__ == '__main__':
    d = orfddataset('/raid/yehongliang_data/ORFD_Dataset_ICRA2022/', mode='train', use_sne=True, use_size=[1280, 704], transform=False)
    for i, data in enumerate(d):
        edge = data['label_edge']
        cv2.imwrite('edge.png', edge * 255)
        # rgb_img = data['rgb_image']
        # cv2.imwrite('rgb_image.png', rgb_img)
        # sn_img = cv2.cvtColor(data['sn_image'], cv2.COLOR_RGB2BGR)
        # cv2.imwrite('sn_image.png', sn_img)
        # label = cv2.resize(data['label'], (1280, 704),interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('label.png', label * 255)
        pass
