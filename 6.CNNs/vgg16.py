#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize VGG16 by pytorch
"""

import sys
import torch
import torch.nn as nn
from collections import OrderedDict
sys.path.append("../d2l_func/")
from model_train import train_pytorch, train_epoch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


def vgg_block(conv_num, in_channels, out_channels):
    """define vgg block because VGG model has the same structure"""
    layer = OrderedDict({})
    for num in range(conv_num):
        if num == 0:
            layer["conv1"] = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=3, padding=1)
#             layer["bn1"] = nn.BatchNorm2d(out_channels)
            layer["relu1"] = nn.ReLU()
        else:
            layer["conv" + str(num + 1)] = nn.Conv2d(out_channels, out_channels,
                                                     kernel_size=3, padding=1)
#             layer["bn" + str(num + 1)] = nn.BatchNorm2d(out_channels)
            layer["relu" + str(num + 1)] = nn.ReLU()
    layer["pool1"] = nn.MaxPool2d(kernel_size=2, stride=2)

    return nn.Sequential(layer)


# define VGG16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            vgg_block(2, 1, 64),
            vgg_block(2, 64, 128),
            vgg_block(2, 128, 256),
            vgg_block(3, 256, 512),
            vgg_block(3, 512, 512),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.shape[0], -1)
        return self.fc(output)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    model = VGG16()
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist(resize=224)

    params = {
        "epoch_num": 2,
        "data_num": len(mnist_train),
        "batch_size": 16,
        "gpu": True,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "draw": True,
        "evaluate": model.score,
        # "accum_step": 8,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"],
                                                    num_workers=8, resize=224)
    params["train_iter"] = train_iter
    params["test_iter"] = test_iter

    # training
    train_epoch(**params)
