#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
sys.path.append("../d2l_func/")
import torch
from model_train import train_experiment
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist
from draw import get_fashion_mnist_label, show_fashion_mnist
import torch.utils.data as Data


class PSoftmaxModel(object):
    """realize softmax model"""
    def __init__(self, fea_num, cate_num):
        self.fea_num = fea_num
        self.cate_num = cate_num
        self.w = torch.normal(0, 0.01, size=(fea_num, cate_num), requires_grad=True)
        self.b = torch.zeros(cate_num, requires_grad=True)

    def linreg(self, X):
        return torch.mm(X.view(-1, self.fea_num), self.w) + self.b

    @staticmethod
    def softmax(y_pred):
        return torch.exp(y_pred)/(torch.exp(y_pred).sum(dim=1, keepdim=True))

    def fit(self, X):
        y_pred = self.softmax(self.linreg(X))
        return y_pred

    @staticmethod
    def entropy_loss(y_pred, y):
        y_pred = torch.gather(y_pred, 1, y.view(-1, 1))
        return -(y_pred.clamp(min=1e-12).log()).sum()/len(y)

    def predict(self, X):
        y_pred = self.fit(X)
        y_pred = torch.argmax(y_pred, dim=1).reshape(-1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = (y_pred == y).sum().item()/len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = PSoftmaxModel(fea_num=28*28, cate_num=10)
    # download fashion_mnist dataset
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "model": model.fit,
        "loss": model.entropy_loss,
        "batch_size": 512,
        "data_num": len(mnist_train),
        "epoch_num": 50,
        "lr": 0.02,
        "params": (model.w, model.b),
        "train_iter": None,
        "test_iter": Data.DataLoader(mnist_test,
                                     batch_size=len(mnist_test), shuffle=True),
        "evaluate": model.score,
        "draw": True,
        "save_fig": True,
    }

    # load fashion mnist
    train_iter, test_iter = load_data_fashion_mnist(params["batch_size"],
                                                    num_workers=8)
    params["train_iter"] = train_iter

    # train
    train_experiment(**params)

    # test
    x, y = iter(test_iter).next()
    true_label = get_fashion_mnist_label(y)
    pred_label = get_fashion_mnist_label(model.predict(x))
    label = [true + "\n" + pred for true, pred in zip(true_label, pred_label)]
    show_fashion_mnist(x[:10], label[:10])
