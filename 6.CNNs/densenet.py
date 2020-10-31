#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize DenseNet by pytorch
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../d2l_func/")
from data_prepare import load_data_fashion_mnist, download_data_fashion_mnist
from model_train import train_epoch


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, 3, padding=1)
    )
    return blk


class DenseBlock(nn.Module):
    """
    function: realize DenseBlock in DenseNet
    params in_channels: the number of channels in input
    params out_channels: Actually, the out_channels stands for the growth rate of concat,
                         when concatenate the conv_block in DenseBlock
    params num_conv: the number of conv layer in a DenseBlock
    """
    def __init__(self, in_channels, out_channels, num_conv):
        super(DenseBlock, self).__init__()
        blk = []
        for num in range(num_conv):
            in_c = in_channels + num * out_channels
            blk.append(conv_block(in_c, out_channels))
        self.block = nn.ModuleList(blk)
        # calculate the number of channels
        self.out_channels = in_channels + num_conv * out_channels

    def forward(self, x):
        for b in self.block:
            y = b(x)
            # concat in channels
            x = torch.cat((x, y), dim=1)
        return x


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        # nn.Conv2d(in_channels, out_channels, 1, stride=2)
        nn.Conv2d(in_channels, out_channels, 1),
        nn.AvgPool2d(2)
    )
    return blk


class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, num_conv_in_dense_block):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("block1", nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        ))
        for i, num_conv in enumerate(num_conv_in_dense_block):
            # dense block
            dense_block = DenseBlock(in_channels, growth_rate, num_conv)
            self.net.add_module("dense_block%d" % (i + 2), dense_block)
            # change the input channels after dense_block
            in_channels = dense_block.out_channels
            # transition block
            if i != (len(num_conv_in_dense_block) - 1):
                self.net.add_module("trans_block%d" % (i + 2),
                                    transition_block(in_channels, in_channels // 2))
                # change the input channels after transition block
                in_channels = in_channels // 2
        # add BN
        self.net.add_module("bn", nn.BatchNorm2d(in_channels))
        self.net.add_module("relu", nn.ReLU())
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())
        self.net.add_module("fc", nn.Sequential(
            FlattenLayer(),
            nn.Linear(in_channels, 10)
        ))

    def forward(self, x):
        return self.net(x)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # model params
    model_params = {
        "in_channels": 64,
        "growth_rate": 32,
        "num_conv_in_dense_block": [4, 4, 4, 4]
    }

    # define model
    model = DenseNet(**model_params)
    model = model.cuda()
    # loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    # training params
    params = {
        "epoch_num": 2,
        "data_num": len(mnist_train),
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "gpu": True,
        "draw": True,
        "batch_size": 64,
        "evaluate": model.score,
    }

    # iterator
    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"],
                                                    num_workers=8, resize=224)
    params["train_iter"] = train_iter
    params["test_iter"] = test_iter

    # training
    train_epoch(**params)
