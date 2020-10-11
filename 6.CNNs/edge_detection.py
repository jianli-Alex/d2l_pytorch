#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize sample edge detection by pytorch
"""

import sys
import torch
from conv_pytorch import Conv2D, conv2d
sys.path.append("../d2l_func/")
from optim import sgd
from sqdm import sqdm


def squared_loss(y_pred, y):
    return ((y_pred - y)**2).sum()


# self-define convolution kernel
k = torch.tensor([[-1, 1]])
# x and calculate true y
x = torch.ones(8, 8)
x[:, 2:6] = 0
y = conv2d(x, k)
print(y)

# train convolution layer to get k
model = Conv2D(k.shape)
loss = squared_loss
epoch_num = 100
lr = 0.01
weight_decay = 0

# training bar
process_bar = sqdm()
# training
for epoch in range(epoch_num):
    print(f"Epoch [{epoch+1}/{epoch_num}]")
    y_pred = model(x)
    l = loss(y_pred, y)

    # bp
    l.backward()
    # update grad
    sgd([model.weight, model.bias], lr, weight_decay)
    # clear grad
    _ = model.weight.grad.fill_(0)
    _ = model.bias.grad.fill_(0)

    process_bar.show_process(1, 1, l.item())
    print("\n")

print(f"true kernel is {k}, training kernel is {model.weight}, bias is {model.bias}")