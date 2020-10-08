#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: simulate fitting poly linear model
"""
import os
import sys
import torch
import pickle
import torch.nn as nn
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_experiment


class PolyModel(nn.Module):
    def __init__(self, fea_num):
        super(PolyModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(fea_num, 1)
        )

    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    # generate origin data
    # true formula: y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + error
    train_num = test_num = 100
    true_w = torch.tensor([1.2, -3.4, 5.6]).view(-1, 1)
    true_b = torch.tensor([5]).view(-1, 1)
    linear_feature = torch.randn(train_num+test_num, 1)
    poly_feature = torch.cat((linear_feature, linear_feature**2,
                              linear_feature**3), dim=1)
    error = torch.normal(0, 0.01, size=(len(poly_feature), 1))
    y = torch.mm(poly_feature, true_w) + true_b + error

    # define model
    model = PolyModel(poly_feature.shape[1])
    loss = nn.MSELoss()
    train_dataset = Data.TensorDataset(poly_feature[:100], y[:100])
    test_dataset = Data.TensorDataset(poly_feature[100:], y[100:])
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
    train_experiment(**params)

    # save data
    if not os.path.exists("./data/"):
        os.mkdir("./data/")
    with open("./data/linear_feature.pkl", "wb+") as f:
        pickle.dump(linear_feature, f)

    with open("./data/poly_feature.pkl", "wb+") as f:
        pickle.dump(poly_feature, f)

    with open("./data/poly_y.pkl", "wb+") as f:
        pickle.dump(y, f)
