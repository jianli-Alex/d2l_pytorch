#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""function: realize dropout model with pytorch (including nn.Module)"""

import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_pytorch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist
from model_class import FlattenLayer


class DropoutModel2(nn.Module):
    def __init__(self):
        super(DropoutModel2, self).__init__()
        self.layer = nn.Sequential(
            FlattenLayer(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layer(x)

    def score(self, x, y):
        y_pred = self.forward(x)
        return (y_pred.argmax(dim=1) == y).sum().item() / len(y)


if __name__ == "__main__":
    # define model
    model = DropoutModel2()
    # change to cuda if cuda is available
    if torch.cuda.is_available():
        model = model.cuda()
    # loss
    loss = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # download data
    train_mnist, test_mnist = download_data_fashion_mnist()

    # parameter
    params = {
        "model": model,
        "loss": loss,
        "epoch_num": 3,
        "data_num": len(train_mnist),
        "batch_size": 512,
        "optimizer": optimizer,
        "test_iter": Data.DataLoader(test_mnist, len(test_mnist), shuffle=True),
        "evaluate": model.score,
        "draw": True,
        "gpu": True,
    }

    # load data
    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"],
                                                    num_workers=8)
    params["train_iter"] = train_iter

    # training
    train_pytorch(**params)