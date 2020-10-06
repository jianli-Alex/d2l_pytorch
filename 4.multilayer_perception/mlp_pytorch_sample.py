#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_pytorch
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist
from draw import get_fashion_mnist_label, show_fashion_mnist


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        return X


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.layer = nn.Sequential(
            FlattenLayer(),
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 10),
        )

    def forward(self, X):
        y = self.layer(X)
        return y

    def score(self, X, y):
        y_pred = self.forward(X)
        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = MLPModel()

    # init weight and bias
    for net in model.layer:
        if isinstance(net, nn.Linear):
            _ = init.normal_(net.weight, 0, 0.01)
            _ = init.constant_(net.bias, 0)

    # loss
    loss = nn.CrossEntropyLoss()
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "model": model,
        "loss": loss,
        "epoch_num": 3,
        "data_num": len(mnist_train),
        "batch_size": 512,
        "lr": 0.1,
        "weight_decay": 0,
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test)),
        "evaluate": model.score,
        "draw": True,
        "save_fig": True,
        # "draw_epoch": True,
    }

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"],
                                weight_decay=params["weight_decay"], momentum=0.9)
    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8)
    params["train_iter"] = train_iter
    params["optimizer"] = optimizer

    # training
    train_pytorch(**params)

    # testing
    test_iter = Data.DataLoader(mnist_test, batch_size=10)
    x, y = iter(test_iter).next()
    true_label = get_fashion_mnist_label(y)
    pred_label = get_fashion_mnist_label(model(x).argmax(dim=1))
    label = [true + "\n" + pred for true, pred in zip(true_label, pred_label)]
    show_fashion_mnist(x, label)
