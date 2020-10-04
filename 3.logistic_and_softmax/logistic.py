#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize logistic model by numpy and test throughout iris dataset in sklearn
"""

import sys
sys.path.append("../d2l_func/")
import numpy as np
from sklearn.datasets import load_iris
from model_selection import bootstrap
from train import *


class LogisticModel(object):
    """realize logisticModel"""
    def __init__(self, alpha=0.01, weight_decay=0):
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.count = 0

    def linreg(self, X):
        """linear model"""
        return X @ self.w + self.b

    def sigmoid(self, y):
        """realize sigmoid"""
        return 1 / (1 + np.exp(-y))

    def entropy_loss(self, y_pred, y):
        """cal entropy loss function which is -mean(ylog(y_pred)+(1-y)log(1-y_pred))"""
        y_pred = np.where(y == 0, 1 - y_pred, y_pred)
        loss = -(np.log(y_pred).sum()) / len(y)
        return loss

    def fit(self, X, y):
        fea_num = int(X.size / len(y))
        if self.count == 0:
            self.w = np.zeros(fea_num)

        # reshape X and y
        X = X.reshape(len(y), fea_num)
        y = y.reshape(-1)

        # predict
        y_pred = self.predict_prob(X)

        # update grad + weight_decay
        dw = (X.T @ (y_pred - y)).sum() / len(y) + self.weight_decay * self.w
        db = (y_pred - y).sum() / len(y) + self.weight_decay * self.b
        self.w -= self.alpha * dw
        self.b -= self.alpha * db
        self.count += 1

    def predict_prob(self, X):
        """calculate probability in each prob"""
        y_pred = self.sigmoid(self.linreg(X)).reshape(-1)
        return y_pred

    def predict(self, X):
        """predict category labels, like 1, 2"""
        y_pred = self.predict_prob(X)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred

    def score(self, X, y):
        """calculate accuracy"""
        y_pred = self.predict(X)
        acc = (y_pred == y).sum() / len(y)
        return acc


# deal with iris_data
iris = load_iris()
iris_data = np.hstack((iris.data, np.expand_dims(iris.target, 1)))
# two category
iris_data = iris_data[iris.target < 2]
data_loader = bootstrap(iris_data[:, :4], iris_data[:, 4])

# training
params={
    "model": LogisticModel(alpha=0.02, weight_decay=0),
    "epoch_num": 100,
    "batch_size": 1,
    "data_loader": data_loader,
}

model = train(**params)
# test result
print(model.predict(iris_data[:, :4]))
print(model.score(iris_data[:, :4], iris_data[:, 4]))
