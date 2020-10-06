#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: simulate the situation with weight decay in a high dimension
linear model by pytorch.
"""

import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_pytorch


class LinearModel(nn.Module):
    def __init__(self, fea_num):
        super(LinearModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(fea_num, 1)
        )

    def forward(self, x):
        return self.layer(x)


# generate data
fea_num, train_num, test_num = 200, 20, 100
true_w = torch.full((fea_num, 1), 0.01)
true_b = torch.tensor([0.05])
x = torch.normal(0, 1, size=(train_num+test_num, fea_num))
error = torch.normal(0, 0.01, size=(len(x), 1))
y = torch.mm(x, true_w) + true_b + error

# define model
model = LinearModel(fea_num)
train_dataset = Data.TensorDataset(x[:20], y[:20])
test_dataset = Data.TensorDataset(x[20:], y[20:])
test_iter = Data.DataLoader(test_dataset,
                            batch_size=len(test_dataset), shuffle=True)
# loss
loss = nn.MSELoss()

params = {
    "model": model,
    "loss": loss,
    "epoch_num": 100,
    "batch_size": 20,
    "lr": 0.01,
    "weight_decay": 2,
    "data_num": 20,
    "test_iter": test_iter,
    "draw": True,
}

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
train_iter = Data.DataLoader(train_dataset, shuffle=True,
                             batch_size=params["batch_size"])
params["optimizer"] = optimizer
params["train_iter"] = train_iter

# training
train_pytorch(**params)