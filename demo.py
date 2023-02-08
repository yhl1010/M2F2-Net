import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import cv2
import numpy as np
import torch
import copy

from models import m2f2_net

import torchvision.transforms as transforms

def main():
    save_path = 'results/'
    os.makedirs(save_path, exist_ok=True)

    net = m2f2_net.mit_b2()

    ckpt_dir = os.path.join('ckpts/', 'test')
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_epoch.pth')))

    net.cuda()
    net.eval()

    useDir = 'ORFD_Dataset_ICRA2022/testing'
    img_file = 'y2021_0228_1802'
    img_name = '1620330240063.png'
    rgb_image_ori = cv2.imread(os.path.join(useDir, img_file, 'image_data', img_name))
    sn_image_ori = cv2.imread(os.path.join(useDir, img_file, 'surface_normal', img_name))

    rgb_image_save = copy.copy(rgb_image_ori)

    oriHeight, oriWidth, _ = rgb_image_ori.shape

    # resize image to enable sizes divide 32
    rgb_image = cv2.resize(rgb_image_ori, (1280, 704))
    sn_image = cv2.resize(sn_image_ori, (1280, 704))

    rgb_image = (rgb_image.astype(np.float32) / 255 - 0.5) / 0.5
    rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0)

    sn_image = (sn_image.astype(np.float32) / 255 - 0.5) / 0.5
    sn_image = transforms.ToTensor()(sn_image).unsqueeze(0)

    with torch.no_grad():
        pred = net(rgb_image.cuda(), sn_image.cuda())
        _, pred = torch.max(pred[0].data.cpu(), 1)
        pred_img = pred[0].cpu().float().detach().numpy()
        pred_img = cv2.resize(pred_img, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

        index = np.where(pred_img == 1)
        rgb_image_save[index[0], index[1], :] = [255, 0, 85]

        img_cat = np.concatenate((rgb_image_ori, sn_image_ori, rgb_image_save), axis=1)
        img_cat = cv2.resize(img_cat, (int(img_cat.shape[1] * 0.5), int(img_cat.shape[0] * 0.5)))

        cv2.imwrite(save_path + 'result_3.png', img_cat)



if __name__ == '__main__':
    main()
