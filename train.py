import argparse
import datetime, time
import os
import cv2
import numpy as np
import random
import shutil

import torch

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from datasets.ORFD_dataset import orfddataset
from models import m2f2_net

from util.criterion import SegLoss
from util.util import confusion_matrix, getScores
from util.lr_adjust import Adjust_learning_rate


start = datetime.datetime.now()


def get_arguments():
    parser = argparse.ArgumentParser(description="ORFD train Network")
    parser.add_argument("--exp_name", type=str, default='test',
                        help="experiment name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_root", type=str, default='/ORFD_Dataset_ICRA2022/',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str, default='1280,704',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lr_policy", type=str, default='poly',
                        help="which lr policy to choose, poly|lambda|cosine")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument('--lr_decay_epochs', type=int, default=25,
                        help='multiply by a gamma every lr_decay_epoch epochs')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                        help='gamma factor for lr_scheduler')
    parser.add_argument('--warm_steps', type=float, default=1000,
                        help='warm steps for cosine policy')

    parser.add_argument("--ckpt_dir", type=str, default='ckpts/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu_ids", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="start epoch")
    parser.add_argument("--epochs", type=int, default=30,
                        help="total epochs")

    return parser.parse_args()


args = get_arguments()


def main():
    save_ckpt_path = os.path.join(args.ckpt_dir, args.exp_name)
    if not os.path.exists(save_ckpt_path):
        os.makedirs(save_ckpt_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    # cudnn.enabled = True
    # # cudnn related setting
    # cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True  # False
    # torch.backends.cudnn.enabled = True
    # torch.cuda.empty_cache()

    net = m2f2_net.mit_b2()
    net.cuda()

    seg_former = torch.load('/raid/yehongliang_data/mit_b2.pth')
    state_dict = seg_former

    new_params = net.state_dict().copy()
    for i in state_dict:
        if 'backbone' in i:
            i_part = i[9:]
        else:
            i_part = i
        new_params[i_part] = state_dict[i]

    net.load_state_dict(new_params, strict=False)

    if args.start_epoch > 0:
        net.load_state_dict(torch.load(os.path.join(save_ckpt_path, 'net_epoch.pth')))

    criterion = SegLoss()
    criterion.cuda()

    lr_adjust = Adjust_learning_rate(args)

    dataset_train = orfddataset(args.data_root, mode='train', use_size=input_size)
    train_num_samples = len(dataset_train)
    print("train data num:", train_num_samples)
    train_dataloader = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dataset_val = orfddataset(args.data_root, mode='val', use_size=input_size)
    val_num_samples = len(dataset_val)
    print("val data num:", val_num_samples)
    val_dataloader = data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4)

    optimizer = optim.AdamW(
        net.parameters(),
        lr = args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    total_iters = args.epochs * len(train_dataloader)
    total_iter_per_batch = len(train_dataloader)
    print("total iters:", total_iters)

    best_f_score = 0
    best_epoch = 0
    temp = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        net.train()
        for i_iter, batch in enumerate(train_dataloader):
            iter_lr = i_iter + epoch * len(train_dataloader)
            lr = lr_adjust.adjust_lr(optimizer=optimizer, iter_lr=iter_lr, total_iters=total_iters, epoch=epoch)

            optimizer.zero_grad()
            pred = net(batch['rgb_image'].cuda(), batch['sn_image'].cuda())
            loss_all, loss_rgb, loss_sn, loss_fuse, loss_edge = criterion(pred, [batch['label'].cuda(), batch['label_edge'].cuda()])
            loss_all.backward()
            optimizer.step()

            if i_iter % 10 == 0:
                tim = time.time()
                print('epoch:{},iter:{}/{},loss_all:{:.3f},loss_rgb:{:.3f},loss_sn:{:.3f},loss_fuse:{:.3f},loss_edge:{:.3f},lr:{:.3e},time:{:.1f}'.
                      format(epoch, i_iter, total_iter_per_batch, loss_all.data.cpu().numpy(), loss_rgb.data.cpu().numpy(),
                             loss_sn.data.cpu().numpy(), loss_fuse.data.cpu().numpy(), loss_edge.data.cpu().numpy(), lr, tim - temp))
                temp = tim

        if epoch % 1 == 0:
            print("----->Epoch:", epoch)
            # torch.save(net.state_dict(), os.path.join(save_ckpt_path, 'epoch_' + str(epoch) + '.pth'))
            acc, pre, recall, F_score, iou = valid(net, val_dataloader)

            is_best_f_score = F_score > best_f_score
            best_f_score = max(F_score, best_f_score)
            ckpt_epoch = os.path.join(save_ckpt_path, 'net_epoch.pth')
            torch.save(net.state_dict(), ckpt_epoch)
            if is_best_f_score:
                best_epoch = epoch
                print("Best F_score epoch: ", epoch)
                best_ckpt_epoch = os.path.join(save_ckpt_path, 'best_epoch.pth')
                shutil.copyfile(ckpt_epoch, best_ckpt_epoch)
            print("best epoch:", best_epoch)
    end = datetime.datetime.now()
    print(end - start, 'seconds')
    print(end)


def valid(model, val_loader):
    model.eval()
    conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    print('valid processing')

    with torch.no_grad():
        for i_iter, batch in enumerate(val_loader):
            preds = model(batch['rgb_image'].cuda(), batch['sn_image'].cuda())
            _, pred = torch.max(preds[0].cpu(), 1)
            pred = pred.float().detach().int().numpy()

            oriSize = (batch['oriSize'][0].numpy()[0], batch['oriSize'][1].numpy()[0])
            gt = np.expand_dims(cv2.resize(np.squeeze(batch['label'].int().numpy(), axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            # gt = np.expand_dims(cv2.resize(batch['label'].int().numpy(), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            # pred = np.expand_dims(cv2.resize(pred, oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            conf_mat += confusion_matrix(gt, pred, args.num_classes)

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)

    print('glob acc:{0:.3f}, pre:{1:.3f}, recall:{2:.3f}, F_score:{3:.3f}, IoU:{4:.3f}'.format(globalacc, pre, recall, F_score, iou))

    return globalacc, pre, recall, F_score, iou


if __name__ == '__main__':
    main()
