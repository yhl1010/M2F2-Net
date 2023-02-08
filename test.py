import os
import cv2
import argparse
import numpy as np
import torch
from torch.utils import data

from datasets.ORFD_dataset import orfddataset
from models import m2f2_net
from util.util import confusion_matrix, getScores


def get_arguments():
    parser = argparse.ArgumentParser(description="ORFD test")
    parser.add_argument("--exp_name", type=str, default='test',
                        help="experiment name")
    parser.add_argument("--gpu_ids", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_root", type=str, default='ORFD_Dataset_ICRA2022/',
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str, default='1280,704',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ckpt_dir", type=str, default='ckpts/',
                        help="Where to save snapshots of the model.")
    return parser.parse_args()


args = get_arguments()


def process():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    save_ckpt_path = os.path.join(args.ckpt_dir, args.exp_name)

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]
    ori_size = (1280, 720)

    dataset_test = orfddataset(args.data_root, mode='test', use_size=input_size)
    test_num_samples = len(dataset_test)
    print("test data num:", test_num_samples)
    test_dataloader = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    net = m2f2_net.mit_b2()
    net.load_state_dict(torch.load(os.path.join(save_ckpt_path, 'best_epoch.pth')))

    net.cuda()
    net.eval()

    conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    print('test processing')

    with torch.no_grad():
        for i_iter, batch in enumerate(test_dataloader):
            pred = net(batch['rgb_image'].cuda(), batch['sn_image'].cuda())
            _, pred = torch.max(pred[0].cpu(), 1)
            pred = pred.float().detach().int().numpy()

            gt = np.expand_dims(cv2.resize(np.squeeze(batch['label'].int().numpy(), axis=0), ori_size, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), ori_size, interpolation=cv2.INTER_NEAREST), axis=0)
            conf_mat += confusion_matrix(gt, pred, args.num_classes)

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)

    print('glob acc:{0:.3f}, pre:{1:.3f}, recall:{2:.3f}, F_score:{3:.3f}, IoU:{4:.3f}'.format(globalacc, pre, recall, F_score, iou))


if __name__ == '__main__':
    process()
