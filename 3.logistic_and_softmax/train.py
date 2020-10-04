#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
sys.path.append("../d2l_func/")
from sqdm import sqdm
from data_prepare import data_iter


def train(data_loader, model, epoch_num, batch_size):
    # generate data
    xtrain, ytrain, xtest, ytest = data_loader

    # training bar
    process_bar = sqdm()

    for epoch in range(epoch_num):
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        for xdata, ydata in data_iter(batch_size, xtrain, ytrain):
            model.fit(xdata, ydata)

            # train
            train_pred = model.predict_prob(xdata)
            train_loss = model.entropy_loss(train_pred,
                                            ydata.reshape(train_pred.shape))
            train_acc = model.score(xdata, ydata)

            # test
            test_pred = model.predict_prob(xtest)
            test_loss = model.entropy_loss(test_pred,
                                           ytest.reshape(test_pred.shape))
            test_acc = model.score(xtest, ytest)

            process_bar.show_process(len(ytrain), batch_size, train_loss=train_loss,
                                     test_loss=test_loss, train_score=train_acc,
                                     test_score=test_acc)

        print("\n")
    return model
