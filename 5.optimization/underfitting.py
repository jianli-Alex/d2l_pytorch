#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: simulate underfitting by using the data in 'polyreg_pytorch.py'
"""
import sys
import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
from polyreg_pytorch import PolyModel
sys.path.append("../d2l_func/")
from model_train import train_pytorch


# load data
with open("./data/linear_feature.pkl", "rb+") as f:
    linear_feature = pickle.load(f)

with open("./data/poly_feature.pkl", "rb+") as f:
    poly_feature = pickle.load(f)

with open("./data/poly_y.pkl", "rb+") as f:
    y = pickle.load(f)

# define model
model = PolyModel(linear_feature.shape[1])
loss = nn.MSELoss()
train_dataset = Data.TensorDataset(linear_feature[:100], y[:100])
test_dataset = Data.TensorDataset(linear_feature[100:], y[100:])
test_iter = Data.DataLoader(test_dataset,
                            batch_size=len(y[:100]), shuffle=True)

params = {
    "model": model,
    "loss": loss,
    "epoch_num": 100,
    "batch_size": 100,
    "lr": 0.01,
    "data_num": 100,
    "test_iter":test_iter,
    "draw": True,
}

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
train_iter = Data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
params["optimizer"] = optimizer
params["train_iter"] = train_iter

# training
train_pytorch(**params)
