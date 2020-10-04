#! /usr/bin/env python
# -*-coding: utf-8 -*-

def squared_loss(y_pred, y):
    """calculate mean square loss without dividing batch_size and 2"""
    return ((y_pred - y)**2).sum()


def sgd2(params, batch_size, lr, weight_decay=0):
    for param in params:
        # param.data = param.data - lr * (param.grad+weight_decay*param.data) / batch_size
        # pytorch practice
        param.data = param.data - lr * (param.grad/batch_size+weight_decay*param.data)


def square_loss(y_pred, y):
    """
    calculate mean square loss which divide batch_size,
    and don't divide batch_size when update gradient by mini-batch GD.
    """
    return ((y_pred - y)**2).sum()/(2*len(y))


def sgd(params, lr, weight_decay=0):
    """realize optimization algorithm """
    for param in params:
        param.data = param.data - lr * (param.grad + weight_decay*param.data)
