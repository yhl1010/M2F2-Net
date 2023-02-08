import torch
import torch.nn as nn
from torch.nn import functional as F


class Edge_Module(nn.Module):
    def __init__(self,in_fea=[128, 128, 128], mid_fea=64, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_fea),
            nn.LeakyReLU(0.2)
            )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_fea),
            nn.LeakyReLU(0.2)
            )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_fea),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Conv2d(mid_fea*3, mid_fea, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(mid_fea, out_fea, kernel_size=1, padding=0)



    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge2_fea = self.conv2(x2)
        edge3_fea = self.conv3(x3)

        edge2_fea =  F.interpolate(edge2_fea, size=(h, w), mode='bilinear',align_corners=True)
        edge3_fea =  F.interpolate(edge3_fea, size=(h, w), mode='bilinear',align_corners=True)

        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge_fea = self.conv4(edge_fea)
        edge = self.conv5(edge_fea)

        return edge, edge_fea


class ASPPModule(nn.Module):
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(inner_features),
                                   nn.LeakyReLU(0.2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1)
            )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle



class SegModule(nn.Module):
    def __init__(self, in_channel=128, mid_channel=64):
        super(SegModule, self).__init__()

        self.aspp_layer = ASPPModule(features=in_channel, inner_features=mid_channel, out_features=mid_channel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, mid_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel*2, mid_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.2)
        )

        self.conv3 = nn.Conv2d(mid_channel, 2, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        _, _, h, w = x1.size()

        x1 = self.conv1(x1)

        x2 = self.aspp_layer(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x = torch.cat([x1, x2], dim=1)
        seg_feat = self.conv2(x)
        seg = self.conv3(seg_feat)

        return seg, seg_feat