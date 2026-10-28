"""Layers used by ADMapNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            mid_channels, mid_channels * 4, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(mid_channels * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


def conv1():
    return nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
    )


def conv2_x():
    layers = []
    downsample = nn.Sequential(
        nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64)
    )
    layers.append(BottleneckBlock(in_channels=16, mid_channels=16, downsample=downsample))
    return nn.Sequential(*layers)


def conv3_x():
    layers = []
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(128)
    )
    layers.append(BottleneckBlock(in_channels=64, mid_channels=32, downsample=downsample))
    return nn.Sequential(*layers)


def conv4_x():
    layers = []
    downsample = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(256)
    )
    layers.append(BottleneckBlock(in_channels=128, mid_channels=64, downsample=downsample))
    return nn.Sequential(*layers)


def conv5_x():
    layers = []
    downsample = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(512)
    )
    layers.append(BottleneckBlock(in_channels=256, mid_channels=128, downsample=downsample))
    return nn.Sequential(*layers)


def global_avg_pool():
    return nn.AdaptiveAvgPool2d((1, 1))


def fc_layer(sources_num):
    fc = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, sources_num))
    return fc
