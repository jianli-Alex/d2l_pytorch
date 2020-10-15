#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize NIN network by pytorch
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../d2l_func/")
from model_train import train_pytorch, train_epoch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 实现nin模块
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )

    return layer


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class NinNet(nn.Module):
    def __init__(self):
        super(NinNet, self).__init__()
        self.layer = nn.Sequential(
            nin_block(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            nin_block(in_channels=384, out_channels=10, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d(),
            FlattenLayer()
        )

    def forward(self, x):
        return self.layer(x)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = NinNet()
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # load data
    mnist_train, mnist_test = download_data_fashion_mnist(resize=224)

    params = {
        "epoch_num": 5,
        "data_num": len(mnist_train),
        "batch_size": 32,
        "gpu": True,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "draw": True,
        "evaluate": model.score,
        "accum_step": 8,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8, resize=224)
    params["train_iter"] = train_iter
    params["test_iter"] = test_iter

    # training
    train_epoch(**params)
