#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize perception in multi-classification task by one vs rest.
In other word, view a category as positive, other categories is negative.
one vs rest will generate c classier, "c" is the number of category.
"""

import numpy as np
from perception_numpy import PerceptionModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class PerceptionOvs(object):
    def __init__(self, alpha=0.001, max_iter_num=1000):
        self.w = None
        self.b = None
        self.y_classes = None
        self.alpha = alpha
        self.max_iter_num = max_iter_num

    def model(self):
        ppn = PerceptionModel(alpha=self.alpha, max_iter_num=self.max_iter_num)
        return ppn

    def linreg(self, X):
        return X@self.w + self.b

    def fit(self, X, y):
        # obtain all category in datasets
        self.y_classes = np.unique(y)
        # init weight and bias
        fea_num = int(X.size / len(y))
        self.w = np.zeros((fea_num, len(self.y_classes)))
        self.b = np.zeros((1, len(self.y_classes)))

        """
        change category matrix, suppose we have three sample which have
        three category [0, 1, 2], we can change to
        --->[[1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]] , view a category as positive and other
            category is negative. Finally, generate 3 classifier.
        """
        y_ovr = np.vstack([np.where(y == i, 1, -1)
                           for i in self.y_classes])

        for index in range(len(y_ovr)):
            ppn = self.model()
            ppn.fit(X, y_ovr[index])
            self.w[:, index] = ppn.w
            self.b[:, index] = ppn.b

    def predict(self, X):
        y_pred = self.linreg(X)
        y_pred = np.argmax(y_pred, axis=1).reshape(-1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = (y_pred == y).sum() / len(y)
        return score


if __name__ == "__main__":
    # load dataset
    iris = load_iris()
    xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target)

    # define model
    model = PerceptionOvs()
    model.fit(xtrain, ytrain)

    # result
    result = model.predict(xtest)
    score = model.score(xtest, ytest)
    all_score = model.score(iris.data, iris.target)
    print(result)
    print(f"the accuracy in test dataset is {score}")
    print(f"the accuracy in all sample is {all_score}")


