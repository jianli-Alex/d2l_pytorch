#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize softmax classify by numpy with iris datasets
"""

import sys
sys.path.append("../d2l_func/")
import numpy as np
from sklearn.datasets import load_iris
from utils import data_iter, bootstrap
from train import *


class SoftmaxModel(object):
    """"realize softmax"""

    def __init__(self, fea_num, cate_num, alpha=0.01, weight_decay=0):
        self.w = np.zeros([fea_num, cate_num])
        self.b = np.zeros(cate_num)
        self.fea_num = fea_num
        self.cate_num = cate_num
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.count = 0

    def linreg(self, X):
        return X @ self.w + self.b

    def softmax(self, y):
        """softmax(z) = exp(z)/sum(exp(z'))"""
        return np.exp(y) / np.expand_dims(np.exp(y).sum(axis=1), 1)

    def entropy_loss(self, y_pred, y):
        """
        loss = mean(y^T log(y_pred)), y and y_pred is a vector about probability
        in each category in a sample. Beside calculate "mean" stand for
        calculate the mean value of entropy loss in all sample
        """
        loss = -(y * np.log(y_pred)).sum() / len(y)
        return loss

    def cal_grad(self, X, y_diff):
        """
        grad = mean(x (y_pred - y)^T), x is a n*p matrix,
        y_pred and y is a n*c matrix, weight which need to update is
        a p*c matrix. Especially, 'p' is the number of feature,
        'n' is the number of sample, 'c' is the number of category
        """
        result = np.zeros([self.fea_num, self.cate_num])
        for i in range(len(X)):
            result += np.outer(X.T[:, i], y_diff[i, :])
        return result / len(X)

    def fit(self, X, y):
        # predict
        y_pred = self.predict_prob(X)

        # update_grad
        dw = self.cal_grad(X, (y_pred - y)) + self.weight_decay * self.w
        db = (y_pred - y).sum(axis=0) / len(y) + self.weight_decay * self.b
        self.w -= self.alpha * dw
        self.b -= self.alpha * db
        self.count += 1

    def predict_prob(self, X):
        """calculate probability in each prob"""
        y_pred = self.softmax(self.linreg(X))
        return y_pred

    def predict(self, X):
        """predict category labels, like 1, 2, 3"""
        y_pred = self.predict_prob(X)
        pred_index = np.argmax(y_pred, axis=1)
        return pred_index

    def score(self, X, y):
        """calculate accuracy"""
        y_pred = self.predict_prob(X)
        pred_index = np.argmax(y_pred, axis=1)
        label_index = np.argmax(y, axis=1)
        acc = (pred_index == label_index).sum() / len(y)
        return acc


# deal with iris dataset
iris = load_iris()
iris_data = np.hstack((iris.data, np.expand_dims(iris.target, 1)))
xtrain, ytrain, xtest, ytest = bootstrap(iris_data[:, :4], iris_data[:, 4])

# deal with label in iris data, such as (0 --> [1, 0, 0])
label_dict = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1]
}
# complete labels change in iris dataset
data = np.array(list(map(lambda x: label_dict[x], iris_data[:, 4])))
# train and test dataset labels change
ytrain = np.array(list(map(lambda x: label_dict[x], list(ytrain))))
ytest = np.array(list(map(lambda x: label_dict[x], list(ytest))))

# train
params={
    "model": SoftmaxModel(fea_num=4, cate_num=3, alpha=0.01, weight_decay=0),
    "epoch_num": 100,
    "batch_size": 1,
    "data_loader": (xtrain, ytrain, xtest, ytest)
}
model = train(**params)
# test
print(model.predict(iris_data[:, :4]))
print(model.score(iris_data[:, :4], data))