#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize AlexNet by pytorch
"""

import sys
import torch
import torch.nn as nn
from collections import OrderedDict
sys.path.append("../d2l_func/")
from model_train import train_pytorch, train_epoch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(OrderedDict({
            "conv1": nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, padding=3, stride=4),
            "relu1": nn.ReLU(),
            "pool1": nn.MaxPool2d(kernel_size=3, stride=2),
            "conv2": nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),
            "relu2": nn.ReLU(),
            "pool2": nn.MaxPool2d(kernel_size=3, stride=2),
            "conv3": nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
            "relu3": nn.ReLU(),
            "conv4": nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            "relu4": nn.ReLU(),
            "conv5": nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            "relu5": nn.ReLU(),
            "pool3": nn.MaxPool2d(kernel_size=3, stride=2),
        }))

        self.fc = nn.Sequential(OrderedDict({
            "fc1": nn.Linear(256 * 6 * 6, 4096),
            "relu1": nn.ReLU(),
            "dropout1": nn.Dropout(0.5),
            "fc2": nn.Linear(4096, 4096),
            "relu2": nn.ReLU(),
            "dropout2": nn.Dropout(0.5),
            "fc3": nn.Linear(4096, 10)
        }))

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(x.shape[0], -1)
        return self.fc(feature)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


# define model
model = AlexNet()
model = model.cuda()
loss = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# load
mnist_train, mnist_test = download_data_fashion_mnist(resize=224)

params = {
    "epoch_num": 5,
    "data_num": len(mnist_train),
    "batch_size": 32,
    "model": model,
    "loss": loss,
    "optimizer": optimizer,
    "gpu": True,
    "draw": True,
    "evaluate": model.score,
    # "accum_step": 8,
    # "draw_mean": True,
}

train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8, resize=224)
params["train_iter"] = train_iter
params["test_iter"] = test_iter

# training
train_epoch(**params)