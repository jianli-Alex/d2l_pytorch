#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize bn by pytorch, but don't use BatchNorm1d and BatchNorm2d
"""

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

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class BatchNorm(nn.Module):
    """
    function: the class of BatchNorm
    params num_features: In fc layer, the num_features stands for the number of features to the next layer,
                         In conv layer, the num_features stands for the the channels of output in the last layer.
    params num_dims: the num_dims stands for the shape of output in the last layer, the fc layer is 2,
                     the conv layer is 4
    """

    def __init__(self, num_features, num_dims, eps=1e-5, momentum=0.9):
        super(BatchNorm, self).__init__()
        assert num_dims in (2, 4)
        if num_dims == 2:
            # fc layer, num_features ---> the number of neural
            shape = (1, num_features)
        else:
            # conv layer, num_features ---> the number of channels
            shape = (1, num_features, 1, 1)

        # init gamma and beta
        # if not use nn.Parametes, it will be a constant parameters which can't learning
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps
        self.momentum = momentum
        self.num_dims = num_dims
        self.mean, self.var, self.x_hat = None, None, None

        # moving mean and var in test
        self.moving_mean, self.moving_var = torch.zeros(shape), torch.zeros(shape)

    def batch_norm(self, x):
        # test mode
        if not self.training:
            self.x_hat = (x - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:
            assert self.num_dims in (2, 4)
            # fc layer
            if self.num_dims == 2:
                self.mean = x.mean(dim=0, keepdim=True)
                # population var in mini-batch
                self.var = ((x - self.mean) ** 2).mean(dim=0, keepdim=True)
            else:
                # conv layer
                self.mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                self.var = ((x - self.mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3,
                                                                                                           keepdim=True)
            # update moving mean and var
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var

            self.x_hat = (x - self.mean) / torch.sqrt(self.var + self.eps)

        # bn
        return self.gamma * self.x_hat + self.beta

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        output = self.batch_norm(x)
        return output


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            BatchNorm(num_features=6, num_dims=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            BatchNorm(num_features=16, num_dims=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5),
            BatchNorm(num_features=120, num_dims=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            BatchNorm(num_features=84, num_dims=2),
            nn.ReLU(),
            #             nn.Dropout(0.5),
            nn.Linear(84, 10),
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
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test),
                                     num_workers=8, pin_memory=True),
        # "save_fig": True,
        # "save_path": "../result/BN对比试验/img/"
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"],
                                                    num_workers=8)
    params["train_iter"] = train_iter

    # training
    train_pytorch(**params)
