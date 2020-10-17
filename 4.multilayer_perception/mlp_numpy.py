#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize multiply perception and backward propagation by number
"""

import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
sys.path.append("../d2l_func/")
from data_prepare import data_iter
from sqdm import sqdm


class MLP(object):
    """
    params alpha: learning rate
    params input_num: the number of input layer
    params output_num: the number of output layer
    params hidden_num: the number of hidden_num, it can accept a/more integer

    for example:
    (0.01, 4, 3, 5)---> the network has one hidden_num which has 5 hidden_unit. Beside, the input layers has
    four unit, the output layers has 3 unit.
    (0.01, 4, 3, 5, 5)---> the network has two hidden_num which has 5 hidden_unit. Beside, the input layers has
    four unit, the output layers has 3 unit.
    """

    def __init__(self, alpha, input_num, output_num, *hidden_num):
        super(MLP, self).__init__()
        self.layer_num = (input_num, *hidden_num, output_num)
        self.weight = {}
        self.bias = {}
        self.dw = {}
        self.db = {}
        self.alpha = alpha
        self.activate = {}
        self.count = 1
        # init weight and bias, the grad of weight and bias
        for idx, num in enumerate(zip(self.layer_num[1:], self.layer_num[:-1])):
            self.weight["w" + str(idx + 1)] = np.random.normal(0, 0.1, size=num)
            self.bias["b" + str(idx + 1)] = np.zeros(num[0])
            self.dw.setdefault("dw" + str(idx + 1), [])
            self.db.setdefault("db" + str(idx + 1), [])

    @staticmethod
    def relu(y_pred):
        return np.maximum(0, y_pred)

    @staticmethod
    def linreg(x, w, b):
        return np.dot(x, w.T) + b

    @staticmethod
    def softmax(y_pred):
        return np.exp(y_pred) / np.expand_dims(np.exp(y_pred).sum(axis=1), 1)

    @staticmethod
    def entropy_loss(y_pred, y):
        return -(y * np.log(y_pred + 1e-8)).sum() / len(y)

    def cal_error_grad(self, y_pred, y):
        """
        calculate the grad of error item in a sample. In particular, the shape of y_pred and y is (1, c),
        the y_pred stands for the output of network, the y stands for the true labels, c stands for
        the number of category.
        """
        theta = -y @ np.diag(1 / (y_pred + 1e-8).reshape(-1)) @ (np.diag(y_pred.reshape(-1)) - y_pred.T @ y_pred)
        return theta.T

    def forward(self, x):
        """
        calculate the activate output in the forward propagation. Beside, if network has one hidden layer,
        it has three activate output. 'a0' is input, 'a1' is the first hidden layer output, 'a2' is the
        network output. Except for the output(like 'a2') using the softmax function, other(not input,
        like 'a1') using the relu function.
        """
        if self.count < len(self.layer_num) - 1:
            output = self.relu(self.linreg(x, self.weight["w" + str(self.count)],
                                           self.bias["b" + str(self.count)]))
            self.activate["a" + str(self.count)] = output
            self.count += 1
            self.forward(output)
        if self.count == len(self.layer_num) - 1:
            a = self.activate["a" + str(self.count - 1)]
            output = self.softmax(self.linreg(a, self.weight["w" + str(self.count)],
                                              self.bias["b" + str(self.count)]))
            self.activate["a" + str(self.count)] = output

        return self.activate["a" + str(self.count)]

    def cal_grad(self, theta, sample_num):
        """
        calculate the grad of wight and bias in each layer.
        dw(l) = theta(l)@[a(l-1)]^T, the a stands for the l-1 layer activate output
        db(l) = theta(l), the theta is the grad of error item in l layer
        theta(l-1) = diag(f'(z(l-1)))@[w(l)]^T@theta(l)
        z(l) = w(l)@a(l-1) + b(l)
        a(l) = f(z(l))
        """
        for i in range(1, len(self.layer_num)):
            num = len(self.layer_num) - i
            act = np.expand_dims(self.activate["a" + str(num - 1)][sample_num], 0)
            self.dw["dw" + str(num)].append(theta @ act)
            self.db["db" + str(num)].append(theta)
            theta = np.diag(np.where(act > 0, 1, 0).reshape(-1)) @ self.weight["w" + str(num)].T @ theta

    def fit(self, x, y):
        # forward propagation
        y_pred = self.predict(x)
        # calculate the grad with a sample in each batch
        for i in range(len(y)):
            # the error item in the last layer
            theta = self.cal_error_grad(np.expand_dims(y_pred[i], 0), np.expand_dims(y[i], 0))
            # cal grad, i stans for the sample which is calculating
            self.cal_grad(theta, i)

        # combine grad with batch
        for i in range(1, len(self.layer_num)):
            num = len(self.layer_num) - i
            self.dw["dw" + str(num)] = sum(self.dw["dw" + str(num)]) / len(y)
            self.weight["w" + str(num)] -= self.alpha * self.dw["dw" + str(num)]
            self.db["db" + str(num)] = sum(self.db["db" + str(num)])
            self.bias["b" + str(num)] -= self.alpha * self.db["db" + str(num)].reshape(-1)

        # clear grad
        for i in range(1, len(self.layer_num)):
            num = len(self.layer_num) - i
            self.dw["dw" + str(num)] = []
            self.db["db" + str(num)] = []

    def predict(self, x):
        self.count = 1
        self.activate["a0"] = x
        y_pred = self.forward(x)
        return y_pred

    def score(self, x, y):
        y_pred = self.predict(x)
        acc = (y_pred.argmax(axis=1) == y).sum() / len(y)
        return acc


if __name__ == "__main__":
    params = {
        "epoch_num": 100,
        "batch_size": 4,
    }

    # deal with label in iris data, such as (0 --> [1, 0, 0])
    label_dict = {
        0: [1, 0, 0],
        1: [0, 1, 0],
        2: [0, 0, 1]
    }

    # load iris data
    iris = load_iris()
    x = iris.data
    y = np.array(list(map(lambda x: label_dict[x], list(iris.target))))

    # define model
    model = MLP(0.03, 4, 3, 5)
    # split dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    # training bar
    process_bar = sqdm()
    for epoch in range(params["epoch_num"]):
        print(f"Epoch [{epoch + 1}/{params['epoch_num']}]")
        for xdata, ydata in data_iter(params["batch_size"], xtrain, ytrain):
            # data fit
            model.fit(xdata, ydata)
            # training
            train_pred = model.predict(xdata)
            train_loss = model.entropy_loss(train_pred, ydata)
            train_acc = model.score(xdata, ydata.argmax(axis=1))

            # test
            test_pred = model.predict(xtest)
            test_loss = model.entropy_loss(test_pred, ytest)
            test_acc = model.score(xtest, ytest.argmax(axis=1))
            process_bar.show_process(len(xtrain), params["batch_size"], train_loss=train_loss,
                                     train_score=train_acc, test_loss=test_loss, test_score=test_acc)

        print("\n")

    print(f"the accuracy of all data: {model.score(x, iris.target)}")

