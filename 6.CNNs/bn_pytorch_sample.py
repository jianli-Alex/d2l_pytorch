#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist
from model_train import train_pytorch


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LeNet5(nn.Module):
    """LeNet5 with BN"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        output = self.conv(x)
        return self.fc(output)

    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = LeNet5()
    model = model.cuda()
    # loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "epoch_num": 2,
        "model": model,
        "loss": loss,
        "data_num": len(mnist_train),
        "optimizer": optimizer,
        "draw": True,
        "gpu": True,
        "batch_size": 256,
        "evaluate": model.score,
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test), num_workers=16, pin_memory=True),
        # "save_fig": True,
        # "save_path": "../result/BN对比试验/img/"
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=16)
    params["train_iter"] = train_iter
    # params["test_iter"] = test_iter

    # training
    train_pytorch(**params)
