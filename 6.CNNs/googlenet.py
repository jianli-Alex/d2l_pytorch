#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
function: realize googleNet by pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../d2l_func/")
from model_train import train_epoch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


class Inception(nn.Module):
    """
    function: realize the Inception model in GoogleNet
    params in_c: the channels of input
    params c1: the channels of the route1
    params c2: the channels of the route2
    params c3: the channels of the route3
    params c4: the channesl of the route4
    """
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.route1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c1, kernel_size=1),
            nn.ReLU(),
        )
        self.route2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.route3 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.route4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_c, out_channels=c4, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        output1 = self.route1(x)
        output2 = self.route2(x)
        output3 = self.route3(x)
        output4 = self.route4(x)
        output = torch.cat((output1, output2, output3, output4), dim=1)
        return output


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            GlobalAvgPool2d()
        )
        self.layer = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            FlattenLayer(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.layer(x)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = GoogleNet()
    model = model.cuda()
    # loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "epoch_num": 2,
        "model": model,
        "data_num": len(mnist_train),
        "loss": loss,
        "optimizer": optimizer,
        "draw": True,
        "gpu": True,
        "batch_size": 64,
        "evaluate": model.score,
        "draw_mean": True,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"] , num_workers=8, resize=224)
    params["train_iter"] = train_iter
    params["test_iter"] = test_iter

    # training
    train_epoch(**params)

