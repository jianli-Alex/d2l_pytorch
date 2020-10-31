#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""function: realize utils in class"""

import torch
import torch.nn as nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

