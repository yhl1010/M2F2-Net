import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(SegLoss, self).__init__()

        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        seg_final = preds[0]
        seg_img_coarse = preds[1]
        seg_sn_coarse = preds[2]
        edge = preds[3]

        seg_label = target[0]
        edge_label = target[1]

        pos_num = torch.sum(edge_label == 1, dtype=torch.float)
        neg_num = torch.sum(edge_label == 0, dtype=torch.float)
        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])

        loss_rgb = self.criterion(seg_img_coarse, seg_label)
        loss_sn = self.criterion(seg_sn_coarse, seg_label)
        loss_fuse = self.criterion(seg_final, seg_label)

        loss_edge = F.cross_entropy(edge, edge_label, weights.cuda(), ignore_index=self.ignore_index)

        loss_all = 0.4 * (loss_rgb + loss_sn) + loss_fuse + 2 * loss_edge

        return loss_all, loss_rgb, loss_sn, loss_fuse, loss_edge
