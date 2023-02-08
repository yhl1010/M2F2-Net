# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize, Upsample
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
from models.decoder_module import Edge_Module, SegModule

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x


class Head(nn.Module):
    def __init__(self, feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512]):
        super(Head, self).__init__()
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = 256

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        decoder_dim=128
        self.edge_layer = Edge_Module(in_fea=[embedding_dim, embedding_dim, embedding_dim], mid_fea=decoder_dim)
        self.seg_layer_img = SegModule(in_channel=embedding_dim, mid_channel=decoder_dim)
        self.seg_layer_sn = SegModule(in_channel=embedding_dim, mid_channel=decoder_dim)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim*3, decoder_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.LeakyReLU(0.2)
        )

        self.seg_pred = nn.Conv2d(decoder_dim, 2, kernel_size=1)
        self.Upsample = Upsample(scale_factor=4)

    def forward(self, inputs, feat_img, feat_sn):
        c1, c2, c3, c4 = inputs

        n, _, h, w = c4.shape

        c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        edge, edge_feat = self.edge_layer(c1, c2, c3)
        seg_img_coarse, seg_img_feat = self.seg_layer_img(feat_img, c4)
        seg_sn_coarse, seg_sn_feat = self.seg_layer_sn(feat_sn, c4)

        fuse_feat = self.conv_fuse(torch.cat([seg_img_feat, seg_sn_feat, edge_feat], dim=1))
        seg_final = self.seg_pred(fuse_feat)

        return seg_final, seg_img_coarse, seg_sn_coarse, edge
