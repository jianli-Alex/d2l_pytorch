#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize perception in multi-classification task by one vs one.
In other word, fit c(c-1)/2 classifier and vote.
the accuracy is usually higher than one vs. rest.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from perceptron_numpy import PerceptronModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class PerceptronOvo(object):
    def __init__(self, alpha=0.001, max_iter_num=1000):
        self.w = None
        self.b = None
        self.y_classes = None
        self.y_combine = None
        self.alpha = alpha
        self.max_iter_num = max_iter_num

    def model(self):
        ppn = PerceptronModel(alpha=self.alpha, max_iter_num=self.max_iter_num)
        return ppn

    def linreg(self, X):
        return X@self.w + self.b

    def fit(self, X, y):
        self.y_classes = np.unique(y)
        self.y_combine = [i for i in combinations(self.y_classes, 2)]

        # init weight and bias
        fea_num = int(X.size / len(y))
        clf_num = len(self.y_combine)
        self.w = np.zeros((fea_num, clf_num))
        self.b = np.zeros((1, clf_num))

        for index, label in enumerate(self.y_combine):
            # choose dataset depend on label
            cond = pd.Series(y).isin(pd.Series(label))
            xdata, ydata = X[cond], y[cond]
            ydata = np.where(ydata == label[0], 1, -1)

            # fit classifier, update w and b
            ppn = self.model()
            ppn.fit(xdata, ydata)
            self.w[:, index] = ppn.w
            self.b[:, index] = ppn.b

    def vote(self, y):
        # voting depend on the result
        y_count = np.unique(y, return_counts=True)
        max_index = np.argmax(y_count[1])
        vote_result = y_count[0][max_index]
        return vote_result

    def predict(self, X):
        y_pred = np.sign(self.linreg(X))
        # reduce label to origin
        for index, label in enumerate(self.y_combine):
            y_pred[:, index] = np.where(y_pred[:, index] == 1, label[0], label[1])
        # extract prediction label in each columns
        predict_zip = zip(*(i.reshape(-1) for i in np.hsplit(y_pred,
                                                             len(self.y_combine))))
        # find max value in each columns by voting
        y_pred = list(map(lambda x: int(self.vote(x)), predict_zip))

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = (y_pred == y).sum() / len(y)
        return acc


if __name__ == "__main__":
    # load dataset
    iris = load_iris()
    xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target)

    # define model
    model = PerceptronOvo()
    model.fit(xtrain, ytrain)

    # result
    result = model.predict(xtest)
    score = model.score(xtest, ytest)
    all_score = model.score(iris.data, iris.target)
    print(result)
    print(f"the accuracy in test dataset is {score}")
    print(f"the accuracy in all sample is {all_score}")