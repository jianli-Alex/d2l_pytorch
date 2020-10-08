#! /usr/bin/env python
# -*-coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append("../d2l_func/")
from sqdm import sqdm
from optim import sgd
from draw import set_fig_display
from decorate import cal_time


@cal_time
def train_experiment(data_num, epoch_num, model, loss, train_iter, batch_size,
                     lr=0.01, weight_decay=0, params=None, optimizer=None,
                     test_iter=None, evaluate=None, draw=False, draw_epoch=False,
                     save_fig=False, save_path="./img/", gpu=False):
    """
    function: training in pytorch (experiment in self-define and nn.Module)
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
    params draw: draw figure with loss and score in train/test whether or not
    params draw_epoch: draw with data in iteration or epoch
    params save_fig: save figure whether or not
    params save_path: the path of saving figure
    params gpu: if want to use gpu and cuda is available, it will send tensor to gpu
    """
    # training bar
    process_bar = sqdm()

    # test data which is used to test after train
    if test_iter is not None:
        test_data, test_label = iter(test_iter).next()
        # if cuda available and want to use gpu
        if torch.cuda.is_available() and gpu:
            test_data = test_data.cuda()
            test_label = test_label.cuda()

    # storage loss and score in train and test
    train_loss_list, test_loss_list = [], []
    train_score_list, test_score_list = [], []
    # iteration num in each epoch
    iter_num = np.ceil(data_num / batch_size)

    for epoch in range(epoch_num):
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        count, mean_train_loss, mean_train_score = 1., 0., 0.
        mean_test_loss, mean_test_score = 0., 0.
        for x, y in train_iter:
            # cuda available and want to use gpu
            if torch.cuda.is_available() and gpu:
                # send tensor from cpu to gpu
                x = x.cuda()
                y = y.cuda()
            # train
            train_pred = model(x)
            train_loss = loss(train_pred, y)
            # calculate mean train loss
            mean_train_loss = (((count - 1) * mean_train_loss +
                                train_loss) / count).item()
            # when True, draw with epoch mean loss in train
            # when False, draw with iteration loss in train (default False)
            if not draw_epoch:
                train_loss_list.append(train_loss.item())
            else:
                if count == iter_num:
                    train_loss_list.append(mean_train_loss)
            # if parameter have criterion(evaluate), like accuracy/f1_score
            # use this criterion to calculate train_score
            if evaluate is not None:
                train_score = evaluate(x, y)
                mean_train_score = ((count - 1) * mean_train_score +
                                    train_score) / count
                # function like the draw_epoch in train loss
                if not draw_epoch:
                    train_score_list.append(train_score)
                else:
                    if count == iter_num:
                        train_score_list.append(mean_train_score)

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

            # test loss
            if test_iter is not None:
                if isinstance(model, nn.Module):
                    # stop dropout and batch normalization
                    model.eval()
                    test_pred = model(test_data)
                    test_loss = loss(test_pred, test_label).item()
                    mean_test_loss = ((count - 1) * mean_test_loss +
                                      test_loss) / count
                    # use this criterion to calculate test_score
                    if evaluate is not None:
                        test_score = evaluate(test_data, test_label)
                        mean_test_score = ((count - 1) * mean_test_score +
                                           test_score) / count
                        if not draw_epoch:
                            test_score_list.append(test_score)
                        else:
                            if count == iter_num:
                                test_score_list.append(mean_test_score)
                    model.train()
                else:
                    # self-define model, if have the parameters "is_training"
                    if "is_training" in model.__code__.co_varnames:
                        test_pred = model(test_data, is_training=False)
                        test_loss = loss(test_pred, test_label).item()
                        mean_test_loss = ((count - 1) * mean_test_loss +
                                          test_loss) / count
                        # use this criterion to calculate test_score
                        if evaluate is not None:
                            test_score = evaluate(test_data, test_label,
                                                  is_training=False)
                            mean_test_score = ((count - 1) * mean_test_score +
                                               test_score) / count
                            if not draw_epoch:
                                test_score_list.append(test_score)
                            else:
                                if count == iter_num:
                                    test_score_list.append(mean_test_score)
                    else:
                        test_pred = model(test_data)
                        test_loss = loss(test_pred, test_label).item()
                        mean_test_loss = ((count - 1) * mean_test_loss +
                                          test_loss) / count
                        # use this criterion to calculate test_score
                        if evaluate is not None:
                            test_score = evaluate(test_data, test_label)
                            mean_test_score = ((count - 1) * mean_test_score +
                                               test_score) / count
                            if not draw_epoch:
                                test_score_list.append(test_score)
                            else:
                                if count == iter_num:
                                    test_score_list.append(mean_test_score)

                # function like the draw_epoch in train loss
                if not draw_epoch:
                    test_loss_list.append(test_loss)
                else:
                    if count == iter_num:
                        test_loss_list.append(mean_test_loss)

            # update counter
            count += 1
            # training bar
            if evaluate is None:
                mean_train_score = "-"
                mean_test_score = "-"
            if test_iter is None:
                mean_test_loss = "-"
                mean_test_score = "-"
            # use mean loss and score in each epoch
            process_bar.show_process(data_num, batch_size=batch_size,
                                     train_loss=mean_train_loss,
                                     train_score=mean_train_score,
                                     test_loss=mean_test_loss,
                                     test_score=mean_test_score)
        print("\n")
    # draw loss figure and score figure
    # especially, loss in train dataset use train_loss not mean_loss in drawing
    if draw:
        # set figure format
        set_fig_display(axes_spines_state=[True] * 4)
        # add new figure and subplot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(len(train_loss_list)), train_loss_list,
                 label="train_loss")
        if len(test_loss_list) > 0:
            ax1.plot(range(len(train_loss_list)), test_loss_list,
                     label="test_loss")
        if draw_epoch:
            ax1.set_xlabel("epoch num")
        else:
            ax1.set_xlabel("iteration num")
        ax1.set_ylabel("loss")

        # ax2
        if evaluate is not None:
            ax2 = ax1.twinx()
            ax2.plot(range(len(train_score_list)), train_score_list, "c-",
                     label="train_score", alpha=0.8)
            if len(test_score_list) > 0:
                ax2.plot(range(len(train_score_list)), test_score_list,
                         "r-", label="test_score", alpha=0.8)
            ax2.set_ylabel("score")
            ax2.set_ylim([0, 1.1])

        if test_iter is None and evaluate is None:
            legend_labels = ["train_loss"]
            legend_loc = [0.75, 0.82]
        elif test_iter is None:
            legend_labels = ["train_loss", "train_score"]
            legend_loc = [0.58, 0.82]
        elif evaluate is None:
            legend_labels = ["train_loss", "test_loss"]
            legend_loc = [0.595, 0.82]
        else:
            legend_labels = ["train_loss", "test_loss",
                             "train_score", "test_score"]
            legend_loc = [0.58, 0.78]
        fig.legend(labels=legend_labels, ncol=2, loc=legend_loc)

        if save_fig:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            fig_name = "fig" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            plt.savefig(save_path + fig_name, dpi=200)

        plt.show()


@cal_time
def train_pytorch(data_num, epoch_num, model, loss, train_iter, batch_size,
                  optimizer=None, test_iter=None, evaluate=None, draw=False,
                  draw_epoch=False, save_fig=False, save_path="./img/", gpu=False):
    """
    function: training in pytorch (only with nn.Module)
    params data_num: the number of sample in train set
    params epoch_num: the number of epoch
    params model: model which fit data
    params loss: calculate loss function
    params train_iter: train data loader
    params optimizer: torch optimizer which is used to update grad
    params test_iter: test data loader, use in testing
    params evaluate: criterion such as acc/f1 score
    params draw: draw figure with loss and score in train/test whether or not
    params draw_epoch: draw with data in iteration or epoch
    params save_fig: save figure whether or not
    params save_path: the path of saving figure
    params gpu: if want to use gpu and cuda is available, it will send tensor to gpu
    """
    # training bar
    process_bar = sqdm()

    # test data which is used to test after train
    if test_iter is not None:
        test_data, test_label = iter(test_iter).next()
        # if cuda available and want to use gpu
        if torch.cuda.is_available() and gpu:
            test_data = test_data.cuda()
            test_label = test_label.cuda()

    # init
    test_loss = test_score = "-"
    # storage loss and score in train and test
    train_loss_list, test_loss_list = [], []
    train_score_list, test_score_list = [], []
    # iteration num in each epoch
    iter_num = np.ceil(data_num / batch_size)

    for epoch in range(epoch_num):
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        count, mean_train_loss, mean_train_score = 1., 0., 0.
        mean_test_loss, mean_test_score = 0., 0.
        for x, y in train_iter:
            # cuda available and want to use gpu
            if torch.cuda.is_available() and gpu:
                # send tensor from cpu to gpu
                x = x.cuda()
                y = y.cuda()
            # train
            # model.train()
            train_pred = model(x)
            train_loss = loss(train_pred, y)
            # calculate mean train loss
            mean_train_loss = (((count - 1) * mean_train_loss +
                                train_loss) / count).item()
            # when True, draw with epoch mean loss in train
            # when False, draw with iteration loss in train (default False)
            if not draw_epoch:
                train_loss_list.append(train_loss.item())
            else:
                if count == iter_num:
                    train_loss_list.append(mean_train_loss)
            # if parameter have criterion(evaluate), like accuracy/f1_score
            # use this criterion to calculate train_score
            if evaluate is not None:
                train_score = evaluate(x, y)
                mean_train_score = ((count - 1) * mean_train_score +
                                    train_score) / count
                # function like the draw_epoch in train loss
                if not draw_epoch:
                    train_score_list.append(train_score)
                else:
                    if count == iter_num:
                        train_score_list.append(mean_train_score)

            # clear grad
            optimizer.zero_grad()
            # bp
            train_loss.backward()
            # grad update
            optimizer.step()

            # test loss
            if test_iter is not None:
                # eval model, it will stop dropout and batch normalization
                model.eval()
                test_pred = model(test_data)
                test_loss = loss(test_pred, test_label).item()
                mean_test_loss = ((count - 1) * mean_test_loss +
                                  test_loss) / count
                # function like the draw_epoch in train loss
                if not draw_epoch:
                    test_loss_list.append(test_loss)
                else:
                    if count == iter_num:
                        test_loss_list.append(mean_test_loss)
                # use this criterion to calculate test_score
                if evaluate is not None:
                    test_score = evaluate(test_data, test_label)
                    mean_test_score = ((count - 1) * mean_test_score +
                                       test_score) / count
                    if not draw_epoch:
                        test_score_list.append(test_score)
                    else:
                        if count == iter_num:
                            test_score_list.append(mean_test_score)
                model.train()

            # update counter
            count += 1
            # training bar
            if evaluate is None:
                mean_train_score = "-"
                mean_test_score = "-"
            if test_iter is None:
                mean_test_loss = "-"
                mean_test_score = "-"
            # use mean loss and score in each epoch
            process_bar.show_process(data_num, batch_size=batch_size,
                                     train_loss=mean_train_loss,
                                     train_score=mean_train_score,
                                     test_loss=mean_test_loss,
                                     test_score=mean_test_score)
        print("\n")

    # draw loss figure and score figure
    # especially, loss in train dataset use train_loss not mean_loss in drawing
    if draw:
        # set figure format
        set_fig_display(axes_spines_state=[True] * 4)
        # add new figure and subplot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(len(train_loss_list)), train_loss_list,
                 label="train_loss")
        if len(test_loss_list) > 0:
            ax1.plot(range(len(train_loss_list)), test_loss_list,
                     label="test_loss")
        if draw_epoch:
            ax1.set_xlabel("epoch num")
        else:
            ax1.set_xlabel("iteration num")
        ax1.set_ylabel("loss")

        # ax2
        if evaluate is not None:
            ax2 = ax1.twinx()
            ax2.plot(range(len(train_score_list)), train_score_list, "c-",
                     label="train_score", alpha=0.8)
            if len(test_score_list) > 0:
                ax2.plot(range(len(train_score_list)), test_score_list,
                         "r-", label="test_score", alpha=0.8)
            ax2.set_ylabel("score")
            ax2.set_ylim([0, 1.1])

        if test_iter is None and evaluate is None:
            legend_labels = ["train_loss"]
            legend_loc = [0.75, 0.82]
        elif test_iter is None:
            legend_labels = ["train_loss", "train_score"]
            legend_loc = [0.58, 0.82]
        elif evaluate is None:
            legend_labels = ["train_loss", "test_loss"]
            legend_loc = [0.595, 0.82]
        else:
            legend_labels = ["train_loss", "test_loss",
                             "train_score", "test_score"]
            legend_loc = [0.58, 0.78]
        fig.legend(labels=legend_labels, ncol=2, loc=legend_loc)

        if save_fig:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            fig_name = "fig" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            plt.savefig(save_path + fig_name, dpi=200)

        plt.show()
