#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize linear model with l2 regularization (bridge regression).
The formula is 'sum((y_pred-y)^2)/2m + lambda*(w^Tw)/2m'. But in pytorch,
nn.MSELoss don't divide 2, and it use l2 regularization to realize weight_decay.
It's noteworthy that w not update with w/m but m in grad update. "m" in this is
standing for sample in each iteration
"""

import sys

sys.path.append("../d2l_func/")
import numpy as np
from sqdm import sqdm
from linear_model import LinearModel
from utils import data_iter


class LinearBridge(LinearModel):
    """Bridge linear model"""

    def __init__(self, alpha=0.01, weight_decay=0.05):
        super(LinearBridge, self).__init__()
        self.alpha = alpha
        self.weight_decay = weight_decay

    def fit(self, X, y):
        # initialize w depend on the X shape
        fea_num = int(X.size / len(y))
        if self.count == 0:
            self.w = np.zeros(fea_num)

        # change X and y shape
        X = X.reshape(len(y), fea_num)
        y = y.reshape(-1)

        # calculate y_pred
        y_pred = self.predict(X)

        # update grad (defined)
        self.w -= self.alpha * (X.T @ (y_pred - y) + self.w * self.weight_decay) / len(y)
        self.b -= self.alpha * ((y_pred - y).sum() + self.b * self.weight_decay) / len(y)
        # update grad (pytorch practice)
        # self.w -= self.alpha * (X.T@(y_pred-y)/len(y) + self.w*self.weight_decay)
        # self.b -= self.alpha * ((y_pred-y).sum()/len(y) + self.b*self.weight_decay)
        self.count += 1


def train(x, y, model, epoch_num, batch_size, alpha, weight_decay):
    model = model(alpha=alpha, weight_decay=weight_decay)
    for epoch in range(epoch_num):
        print(f"Epoch [{epoch+1}/{epoch_num}]")
        for xdata, ydata in data_iter(batch_size, x, y):
            model.fit(xdata, ydata)
            mse = model.score(xdata, ydata)
            process_bar.show_process(len(y), batch_size, mse)
        print("\n")
    return model


"""generate data"""
input_num = 10000
true_w = np.array([2, -3.2])
true_b = np.array([4.2])
x = np.random.normal(0, 1, size=(input_num, len(true_w)))
error = np.random.normal(0, 0.01, size=input_num)
y = x @ true_w + true_b + error

"""model training"""
params = {
    "x": x,
    "y": y,
    "epoch_num": 100,
    "model": LinearBridge,
    "batch_size": 128,
    "alpha": 0.01,
    "weight_decay": 0.05,
}
process_bar = sqdm()
model = train(**params)
print(f"w before update is {true_w}, w after update is {model.w}")
print(f"b before update is {true_b}, b after update is {model.b}")
