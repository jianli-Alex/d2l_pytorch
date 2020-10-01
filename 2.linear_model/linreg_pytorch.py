#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
sys.path.append("../d2l_func/")
import torch
from sqdm import sqdm
from utils import *


def linreg(X, w, b):
    """realize linear model"""
    return torch.mv(X, w) + b


def square_loss(y_pred, y):
    """
    calculate mean square loss which divide batch_size,
    and don't divide batch_size when update gradient by mini-batch GD.
    """
    return ((y_pred - y)**2).sum()/(2*len(y))


def sgd(params, lr):
    """realize optimization algorithm """
    for param in params:
        param.data -= lr * param.grad


def train(epoch_num, net, loss, batch_size, lr):
    """train function"""
    for epoch in range(epoch_num):
        print(f"Epoch [{epoch}/{epoch_num}]")
        for xdata, ydata in data_iter(batch_size, x, y):
            l = loss(net(xdata, w, b), ydata)
            l.backward()
            sgd([w, b], lr)

            # clear grad, aviod grad accumulate
            w.grad.data.zero_()
            b.grad.data.zero_()

            # training bar
            mse = np.round(loss(net(xdata, w, b), ydata).item(), 5)
            process_bar.show_process(len(y), batch_size, mse)
        print("\n")


"""generate data by pytorch"""
input_num = 10000
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
x = torch.normal(mean=0, std=1, size=(input_num, len(true_w)))
error = torch.normal(mean=0, std=0.01, size=(input_num, ))
y = torch.mv(x, true_w) + true_b + error

"""training"""
# set parameter
params = {
    "net": linreg,
    "loss": square_loss,
    "epoch_num": 20,
    "batch_size": 128,
    "lr": 0.01
}

# weight and bias initialize
w = torch.normal(mean=0, std=0.01, size=(2, ), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
process_bar = sqdm()
train(**params)
print(f"w before update is {true_w}, w after update is {w}")
print(f"b before update is {true_b}, b after update is {b}")
