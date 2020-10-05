#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_pytorch
from draw import get_fashion_mnist_label, show_fashion_mnist
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # fashion_mnist is 1*28*28, change to 28*28
        return x.view(x.shape[0], -1)


class PSoftmaxModel2(nn.Module):
    """realize softmax by pytorch"""
    def __init__(self, fea_num, cate_num):
        super(PSoftmaxModel2, self).__init__()
        self.layer = nn.Sequential(
            FlattenLayer(),
            nn.Linear(fea_num, cate_num),
        )

    def forward(self, X):
        """
        nn.CrossEntropyLoss equal to softmax+entropy_loss,
        so the PsoftmaxModel don't add softmax operation
        """
        return self.layer(X)

    def score(self, x, y):
        # softmax don't change the relative value and rank
        y_pred = self.forward(x)
        return (torch.argmax(y_pred, dim=1) == y).sum().item() / len(y)


if __name__ == "__main__":
    # define model
    model = PSoftmaxModel2(fea_num=28*28, cate_num=10)
    # initialize weight and bias
    init.normal_(model.layer[1].weight, 0, 0.01)
    init.constant_(model.layer[1].bias, 0)
    # loss function
    loss = nn.CrossEntropyLoss()
    # download fashion_mnist dataset
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "model": model,
        "loss": loss,
        "batch_size": 512,
        "data_num": len(mnist_train),
        "epoch_num": 50,
        "lr": 0.01,
        "weight_decay": 0,
        "optimizer": None,
        "train_iter": None,
        "test_iter": Data.DataLoader(mnist_test,
                                     batch_size=len(mnist_test), shuffle=True),
        "evaluate": model.score,
    }

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9,
                                weight_decay=params["weight_decay"])
    # load fashion mnist
    train_iter, test_iter = load_data_fashion_mnist(params["batch_size"],
                                                    num_workers=8)
    params["train_iter"] = train_iter
    params["optimizer"] = optimizer

    # train
    train_pytorch(**params)

    # test
    x, y = iter(test_iter).next()
    true_label = get_fashion_mnist_label(y)
    pred_label = get_fashion_mnist_label(torch.argmax(model(x), dim=1))
    label = [true + "\n" + pred for true, pred in zip(true_label, pred_label)]
    show_fashion_mnist(x[:10], label[:10])
