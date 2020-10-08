#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import torch
import torch.utils.data as Data
sys.path.append("../d2l_func/")
from model_train import train_experiment
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist
from draw import get_fashion_mnist_label, show_fashion_mnist


class MLP(object):
    def __init__(self, input_num, hidden_num, output_num):
        self.w1 = torch.normal(0, 0.01, size=(input_num, hidden_num), requires_grad=True)
        self.b1 = torch.zeros(hidden_num, requires_grad=True)
        self.w2 = torch.normal(0, 0.01, size=(hidden_num, output_num), requires_grad=True)
        self.b2 = torch.zeros(output_num, requires_grad=True)
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

    @staticmethod
    def linreg(X, w, b):
        return torch.mm(X, w) + b

    @staticmethod
    def relu(y_pred):
        # pytorch中没有maximum
        return torch.max(y_pred, torch.tensor([0.]))

    @staticmethod
    def softmax(y_pred):
        return torch.exp(y_pred) / (torch.exp(y_pred).sum(axis=1, keepdims=True))

    def entropy_loss(self, y_pred, y):
        y_pred = torch.gather(y_pred, 1, y.unsqueeze(1))
        return -(y_pred.clamp(1e-12).log()).sum() / len(y)

    def predict_prob(self, X):
        # reshape X
        X = X.view(-1, self.input_num)
        a1 = self.relu(self.linreg(X, self.w1, self.b1))
        y_pred = self.softmax(self.linreg(a1, self.w2, self.b2))
        return y_pred

    def predict(self, X):
        y_pred = self.predict_prob(X)
        y_pred = torch.argmax(y_pred, dim=1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = (y_pred == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = MLP(input_num=28 * 28, hidden_num=300, output_num=10)
    # load data
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "model": model.predict_prob,
        "loss": model.entropy_loss,
        "epoch_num": 3,
        "data_num": len(mnist_train),
        "batch_size": 512,
        "lr": 0.1,
        "weight_decay": 0,
        "params": [model.w1, model.b1, model.w2, model.b2],
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test)),
        "evaluate": model.score,
        "draw": True,
        "draw_epoch": True,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8)
    params["train_iter"] = train_iter
    train_experiment(**params)

    # testing
    x, y = iter(test_iter).next()
    true_label = get_fashion_mnist_label(y)
    pred_label = get_fashion_mnist_label(model.predict(x))
    label = [true + "\n" + pred for true, pred in zip(true_label, pred_label)]
    show_fashion_mnist(x[:10], label[:10])
