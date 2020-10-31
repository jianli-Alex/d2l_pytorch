#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize resnet
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
sys.path.append("../d2l_func/")
from data_prepare import load_data_fashion_mnist, download_data_fashion_mnist
from model_train import train_epoch


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    """
    function: realize the Residual Module
    """
    def __init__(self, in_channels, out_channels, use_1x1=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        # if use_1x1 conv, we define a 1x1 conv
        if use_1x1:
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )
        else:
            self.conv3 = None
        # if not use 1x1 conv, add relu after conv2. otherwise, add relu after conv3.
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        # use 1x1 conv
        if self.conv3:
            x = self.conv3(x)
        return self.relu(output + x)


def residual_block(in_channels, out_channels, res_num, first_block=False):
    """
    function: realize residual_block which has two Residual Module
    params in_channels: the channels of input
    params out_channels: the channels of output
    params res_num: the number of residual module
    params first_block: if the first block, the in_channels is equal to out_channels
    """
    blk = []
    if first_block:
        assert in_channels == out_channels
        
    for num in range(res_num):
        # Except the first block, the first Residual in each block use 1x1 conv, stride=2
        if num == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
        
    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            "block1": nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # overlapping max pool
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            
            "block2": residual_block(64, 64, 2, first_block=True),
            "block3": residual_block(64, 128, 2),
            "block4": residual_block(128, 256, 2),
            "block5": residual_block(256, 512, 2),
            # output shape (Batch, 512, 1, 1)
            "global_avg_pool": GlobalAvgPool2d(),
            "fc": nn.Sequential(
                FlattenLayer(),
                nn.Linear(512, 10)
            )
        }))
        
    def forward(self, x):
        return self.net(x)
    
    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = ResNet()
    model = model.cuda()
    # loss
    loss = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    # params
    params = {
        "epoch_num": 2,
        "data_num": len(mnist_train),
        "model": model,
        "loss": loss,
        "batch_size": 64,
        "optimizer": optimizer,
        "evaluate": model.score,
        "gpu": True,
        "draw": True,
        "save_fig": True,
    }
    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8, resize=224)
    params["train_iter"] = train_iter
    params["test_iter"] = test_iter

    # training
    train_epoch(**params)
