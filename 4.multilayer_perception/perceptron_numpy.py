#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize perception with numpy
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class PerceptronModel(object):
    def __init__(self, alpha=0.001, max_iter_num=1000):
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.max_iter_num = max_iter_num
        self.count = 1

    def linreg(self, X):
        return X @ self.w + self.b

    def fit(self, X, y):
        # init weight
        fea_num = int(X.size / len(y))
        if self.count == 1:
            self.w = np.zeros(fea_num)

        # reshape x and y
        X = X.reshape(len(y), fea_num)
        y = y.reshape(-1)

        # update statement
        state = self.predict(X) != y

        # update grad by sgd with error point
        while (state.any()) and (self.count <= self.max_iter_num):
            """
            dw = -yi*xi, db = -yi
            use the grad of error point to update
            """
            self.w += self.alpha * y[state][0] * X[state][0]
            self.b += self.alpha * y[state][0]
            self.count += 1
            state = self.predict(X) != y

        print(f"fit PerceptronModel(alpha = {self.alpha}, "
              f"max_iter_epoch = {self.max_iter_num}, "
              f"total_iter_epoch = {min(self.count, self.max_iter_num)})")

    def predict(self, X):
        y_pred = np.sign(self.linreg(X))
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = (y_pred == y).sum() / len(y)
        print(acc)
        return acc


if __name__ == "__main__":
    # load iris
    iris = load_iris()
    iris_data = np.hstack((iris.data, np.expand_dims(iris.target, 1)))
    iris_data = iris_data[iris.target < 2]
    iris_data = np.where(iris_data == 0, -1, iris_data)
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data[:, :4],
                                                    iris_data[:, 4])

    # define model and fit
    model = PerceptronModel()
    model.fit(xtrain, ytrain)
    # predict
    result = model.predict(xtest)
    score = model.score(xtest, ytest)
    print(result)
    print(score)
