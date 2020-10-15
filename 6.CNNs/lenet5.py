#!/usr/env/bin python
# -*-coding: utf-8 -*-


"""
function: realize lenet5 by pytorch
"""

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
from collections import OrderedDict
sys.path.append("../d2l_func/")
from model_train import train_pytorch, train_epoch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        # output shape is (batch, 120, 1, 1), change it to vector
        output = self.conv(x)
        output = output.view(output.shape[0], -1)
        return self.fc(output)
    
    def score(self, x, y):
        y_pred = self.forward(x)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model(change to gpu) and loss
    model = LeNet5()
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # load
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "epoch_num": 50,
        "data_num": len(mnist_train),
        "batch_size": 512,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "gpu": True,
        "draw": True,
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True),
        "evaluate": model.score,
        "draw_mean": True,
        "save_fig": True,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8)
    params["train_iter"] = train_iter

    # training
    train_epoch(**params)
