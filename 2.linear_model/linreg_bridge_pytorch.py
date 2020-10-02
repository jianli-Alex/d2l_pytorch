#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize bridge linear model by pytorch (don't use nn.Sequential)
"""

import sys
sys.path.append("../d2l_func/")
import torch
from sqdm import sqdm
from utils import data_iter


def squared_loss(y_pred, y):
    """calculate mean square loss without dividing batch_size and 2"""
    return ((y_pred - y)**2).sum()


def sgd2(params, batch_size, lr, weight_decay):
    for param in params:
        # param.data = param.data - lr * (param.grad+weight_decay*param.data) / batch_size
        # pytorch practice
        param.data = param.data - lr * (param.grad/batch_size+weight_decay*param.data)


def linreg(x, w, b):
    return torch.mv(x, w) + b


"""generate data, set random seed which is used tocompare with pytorch with framework"""
input_num = 10000
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])

torch.manual_seed(input_num)
x = torch.normal(0, 1, size=(input_num, len(true_w)))
error = torch.normal(0, 0.01, size=(input_num, ))
y = torch.mv(x, true_w) + true_b + error

"""model training"""
params = {
    "epoch_num": 10,
    "lr": 0.01,
    "weight_decay": 0.05,
    "batch_size": 128
}
# parameter init
torch.manual_seed(100)
w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# training bar
process_bar = sqdm()

for epoch in range(params["epoch_num"]):
    print(f"Epoch [{epoch}/{params['epoch_num']}]")
    for xdata, ydata in data_iter(params["batch_size"], x, y):
        y_pred = linreg(xdata, w, b)
        l = squared_loss(y_pred, ydata.reshape(y_pred.shape))
        l.backward()
        sgd2([w, b], len(ydata), params["lr"], params["weight_decay"])

        # clear grad, aviod grad accumulate
        _ = w.grad.data.zero_()
        _ = b.grad.data.zero_()

        process_bar.show_process(input_num, params["batch_size"], round(l.item(), 5))
    print("\n")

print(f"w before update is {true_w}, w after update is {w}")
print(f"b before update is {true_b}, b after update is {b}")


