#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
sys.path.append("../d2l_func/")
from sqdm import sqdm
from optim import sgd


def train_pytorch(data_num, epoch_num, model, loss, train_iter, batch_size,
                  lr=0.01, weight_decay=0, params=None, optimizer=None,
                  test_iter=None, evaluate=None):
    """
    function: training in pytorch
    params data_num: the number of sample in train set
    params epoch_num: the number of epoch
    params model: model which fit data
    params loss: calculate loss function
    params train_iter: train data loader
    params lr: learning rate
    params weight_decay: the wight of weight_decay/L2 regularization
    params params: the parameters needed to update grad
    params optimizer: torch optimizer which is used to update grad
    params test_iter: test data loader, use in testing
    params evaluate: criterion such as acc/f1 score
    """
    # training bar
    process_bar = sqdm()

    if test_iter is not None:
        test_data, test_label = iter(test_iter).next()

    for epoch in range(epoch_num):
        print(f"Epoch [{epoch+1}/{epoch_num}]")
        count, mean_loss, mean_score = 1., 0., 0.
        for x, y in train_iter:
            # train
            train_pred = model(x)
            train_loss = loss(train_pred, y)
            train_score = evaluate(x, y)

            # clear grad
            if optimizer is not None:
                # use torch optimizer
                optimizer.zero_grad()
            elif (params is not None) and (params[0].grad is not None):
                # the grad is None when we define ourself and use in first time
                for param in params:
                    param.grad.data.zero_()

            # bp
            train_loss.backward()
            # grad update
            if optimizer is not None:
                optimizer.step()
            else:
                sgd(params, lr, weight_decay)

            # calculate mean train loss
            mean_loss = ((count-1)*mean_loss + train_loss)/count
            mean_score = ((count-1)*mean_score + train_score)/count

            # test loss
            if test_iter is not None:
                test_pred = model(test_data)
                test_loss = loss(test_pred, test_label)
                test_score = evaluate(test_data, test_label)
            # training bar
            process_bar.show_process(data_num, batch_size=batch_size,
                                     train_loss=mean_loss.item(),
                                     train_score=mean_score,
                                     test_loss=test_loss.item(),
                                     test_score=test_score)
        print("\n")