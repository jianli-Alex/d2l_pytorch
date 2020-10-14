#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize dropout with pytorch (without nn)
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.utils.data as Data
import sys
sys.path.append("../d2l_func/")
from model_train import train_experiment
from data_prepare import download_data_fashion_mnist, load_data_fashion_mnist


def cal_time(func):
    """calculate execute time"""
    def now_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"execute time is {end - start} seconds")
    return now_time

class DropoutModel1(object):
    """
    if cuda is available and want to use gpu to speed up, it's a good idea to
    set gpu is True
    """
    def __init__(self, input_num, hidden1_num, hidden2_num, output_num, gpu=False):
        if torch.cuda.is_available() and gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.w1 = torch.normal(0, 0.01, size=(input_num, hidden1_num), requires_grad=True, device=self.device)
        self.b1 = torch.zeros(256, requires_grad=True, device=self.device)
        self.w2 = torch.normal(0, 0.01, size=(hidden1_num, hidden2_num), requires_grad=True, device=self.device)
        self.b2 = torch.zeros(256, requires_grad=True, device=self.device)
        self.w3 = torch.normal(0, 0.01, size=(hidden2_num, output_num), requires_grad=True, device=self.device)
        self.b3 = torch.zeros(10, requires_grad=True, device=self.device)

    @staticmethod
    def linreg(x, w, b):
        return torch.mm(x, w) + b

    @staticmethod
    def softmax(y_pred):
        return torch.exp(y_pred) / (torch.exp(y_pred).sum(dim=1, keepdim=True))

    def entropy_loss(self, y_pred, y):
        y_pred = torch.gather(y_pred, 1, y.unsqueeze(1))
        return -(y_pred.clamp(1e-12).log()).sum() / len(y)

    def relu(self, y_pred):
        return torch.max(torch.tensor([0.], device=self.device), y_pred)

    @staticmethod
    def dropout(x, drop_prob):
        # drop_prob是丢弃概率
        x = x.float()
        assert 0 <= drop_prob <= 1
        if drop_prob == 1:
            return torch.zeros_like(x)
        mask = (torch.rand_like(x) > drop_prob).float()
        return mask * x / (1 - drop_prob)

    def predict_prob(self, x, is_training=True):
        x = x.view(x.shape[0], -1)
        y_pred = self.relu(self.linreg(x, self.w1, self.b1))
        if is_training:
            y_pred = self.dropout(y_pred, 0.2)
        y_pred = self.relu(self.linreg(y_pred, self.w2, self.b2))
        if is_training:
            y_pred = self.dropout(y_pred, 0.5)
        return self.softmax(self.linreg(y_pred, self.w3, self.b3))

    def predict(self, x, is_training=True):
        y_pred = self.predict_prob(x, is_training=is_training)
        return y_pred.argmax(dim=1).reshape(-1)

    def score(self, x, y, is_training=True):
        y_pred = self.predict(x, is_training=is_training)
        acc = (y_pred == y).sum().item() / len(y)
        return acc


if __name__ == "__main__":
    # define model
    model = DropoutModel1(28 * 28, 256, 256, 10, gpu=True)
    # load dataset
    mnist_train, mnist_test = download_data_fashion_mnist()

    params = {
        "model": model.predict_prob,
        "loss": model.entropy_loss,
        "epoch_num": 10,
        "lr": 0.1,
        "data_num": len(mnist_train),
        "batch_size": 512,
        "params": [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3],
        "test_iter": Data.DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=True),
        "evaluate": model.score,
        "draw": True,
        # "save_fig": True,
        "gpu": True,
        "draw_epoch": True,
    }

    train_iter, test_iter = load_data_fashion_mnist(batch_size=params["batch_size"], num_workers=8)
    params["train_iter"] = train_iter

    # training
    train_experiment(**params)