#! /usr/bin/env python
# -*-coding: utf-8 -*-

import numpy as np


class LinearModel(object):
    """
    function：training linear model by mini_batch GD
    params alpha：learning rate
    """
    def __init__(self, alpha=0.01):
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.count = 0

    def linreg(self, X):
        return X@self.w + self.b

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

        # update grad
        self.w -= self.alpha * (X.T @ (y_pred - y)) / len(y)
        self.b -= self.alpha * (y_pred - y).sum() / len(y)
        self.count += 1

    def predict(self, X):
        y_pred = self.linreg(X)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = ((y_pred - y)**2).sum()
        return mse