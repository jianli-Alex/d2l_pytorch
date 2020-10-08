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

