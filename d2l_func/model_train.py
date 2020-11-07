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
from optim import sgd, grad_clipping
from draw import set_fig_display
from decorate import cal_time
from data_prepare import to_onehot


def loss_error_sample(list_length, sample_rate):
    """
    function: sample with list, return sample index
    """
    index_array = np.arange(list_length)
    index = np.random.choice(index_array, replace=False,
                             size=int(list_length * sample_rate))
    index.sort()
    return list(index)


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
            ax2.set_ylim([0, 1.18])
            legend_labels = ["train_loss", "test_loss",
                             "train_score", "test_score"]
            legend_loc = [0.58, 0.77]
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
                  draw_epoch=False, save_fig=False, save_path="./img/",
                  gpu=False, sample_rate=1, accum_step=1):
    """
    function: training in pytorch (only with nn.Module), when you choose (draw_epoch false),
              we draw with loss and score in each iteration. when you choose
              (draw_epoch True), we draw with mean loss and score in each epoch.
              But in terms of the number show in training bar, such as train_loss,
              test_loss..., we always show the mean loss and score.
    params data_num: the number of sample in train set
    params epoch_num: the number of epoch
    params model: model which fit data
    params loss: calculate loss function
    params train_iter: train data loader
    params batch_size: the size of each batch_size
    params optimizer: torch optimizer which is used to update grad
    params test_iter: test data loader, use in testing
    params evaluate: criterion such as acc/f1 score
    params draw: draw figure with loss and score in train/test whether or not
    params draw_epoch: draw with data in iteration or epoch
    params save_fig: save figure whether or not
    params save_path: the path of saving figure
    params gpu: if want to use gpu and cuda is available, it will send tensor to gpu
    params sample_rate: the rate of sample (default 1), when draw in iteration, it will
                        show the dense figure. Thus, you can sample some point to draw.
                        the value of sample_rate is between 0-1
    params accum_step: use in gradient accumulation, after the number of accum_step,
                        it will be update the grad of parameters
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
            model.train()
            train_pred = model(x)
            train_loss = loss(train_pred, y) / accum_step
            # calculate mean train loss
            mean_train_loss = (((count - 1) * mean_train_loss +
                                accum_step * train_loss) / count).item()
            # when True, draw with epoch mean loss in train
            # because we train in a batch_size, so we want to show the loss in complete train
            # when False, draw with iteration loss in train (default False)
            if not draw_epoch:
                train_loss_list.append(accum_step * train_loss.item())
            else:
                if count == iter_num:
                    train_loss_list.append(mean_train_loss)
            # if parameter have criterion(evaluate), like accuracy/f1_score
            # use this criterion to calculate train_score
            model.eval()
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
            model.train()

            # bp
            train_loss.backward()
            if count % accum_step == 0:
                # grad update
                optimizer.step()
                # clear grad
                optimizer.zero_grad()

            if count == iter_num:
                optimizer.zero_grad()

            # test loss
            with torch.no_grad():
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
    sample_index = loss_error_sample(len(train_loss_list), sample_rate=sample_rate)
    if draw:
        # set figure format
        set_fig_display(axes_spines_state=[True] * 4)
        # add new figure and subplot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # sample
        train_loss_list = np.array(train_loss_list)[sample_index]
        ax1.plot(range(len(train_loss_list)), train_loss_list,
                 label="train_loss")
        if len(test_loss_list) > 0:
            # sample
            test_loss_list = np.array(test_loss_list)[sample_index]
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
            # sample
            train_score_list = np.array(train_score_list)[sample_index]
            ax2.plot(range(len(train_score_list)), train_score_list, "c-",
                     label="train_score", alpha=0.8)
            if len(test_score_list) > 0:
                # sample
                test_score_list = np.array(test_score_list)[sample_index]
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
            ax2.set_ylim([0, 1.18])
            legend_labels = ["train_loss", "test_loss",
                             "train_score", "test_score"]
            legend_loc = [0.58, 0.77]
        fig.legend(labels=legend_labels, ncol=2, loc=legend_loc)

        if save_fig:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            fig_name = "fig" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            plt.savefig(save_path + fig_name, dpi=200)

        plt.show()


@cal_time
def train_epoch(data_num, epoch_num, model, loss, train_iter, batch_size,
                optimizer=None, test_iter=None, evaluate=None, draw=False,
                draw_mean=False, save_fig=False, save_path="./img/",
                accum_step=1, gpu=False):
    """
    function: training in pytorch (only with epoch), it will speed up in training in epoch
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
    params accum_step: use in gradient accumulation, after the number of accum_step,
                        it will be update the grad of parameters
    params gpu: if want to use gpu and cuda is available, it will send tensor to gpu
    """
    # training bar
    process_bar = sqdm()

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
        test_num, mean_test_loss, mean_test_score = 0., 0., 0.
        for x, y in train_iter:
            # cuda available and want to use gpu
            if torch.cuda.is_available() and gpu:
                # send tensor from cpu to gpu
                x = x.cuda()
                y = y.cuda()
            # train
            model.train()
            train_pred = model(x)
            train_loss = loss(train_pred, y) / accum_step
            # calculate mean train loss
            mean_train_loss = (((count - 1) * mean_train_loss +
                                accum_step * train_loss) / count).item()
            # when draw_mean is True, we save the mean_train_loss in each iteration,
            # otherwise, we save the train_loss in the last iteration in each epoch
            if count == iter_num:
                if draw_mean:
                    train_loss_list.append(mean_train_loss)
                else:
                    train_loss_list.append(accum_step * train_loss)
            # if parameter have criterion(evaluate), like accuracy/f1_score
            # use this criterion to calculate train_score
            model.eval()
            if evaluate is not None:
                train_score = evaluate(x, y)
                mean_train_score = ((count - 1) * mean_train_score +
                                    train_score) / count
                # function like the draw_epoch in train loss
                if count == iter_num:
                    if draw_mean:
                        train_score_list.append(mean_train_score)
                    else:
                        train_score_list.append(train_score)
            model.train()

            # bp
            train_loss.backward()
            if (count % accum_step) == 0:
                # grad update
                optimizer.step()
                # clear grad
                optimizer.zero_grad()
            if count == iter_num:
                optimizer.zero_grad()

            # test loss
            with torch.no_grad():
                if (test_iter is not None) and (count == iter_num):
                    # eval model, it will stop dropout and batch normalization
                    model.eval()
                    for test_data, test_label in test_iter:
                        # calculate the num of test set
                        test_num += len(test_label)
                        # if cuda available and want to use gpu
                        if torch.cuda.is_available() and gpu:
                            test_data = test_data.cuda()
                            test_label = test_label.cuda()

                        test_pred = model(test_data)
                        test_loss = loss(test_pred, test_label).item()
                        mean_test_loss += len(test_label) * test_loss
                        # print(mean_test_loss)
                        # use this criterion to calculate test_score
                        if evaluate is not None:
                            test_score = evaluate(test_data, test_label)
                            mean_test_score += len(test_label) * test_score
                    # result
                    mean_test_loss /= test_num
                    test_loss_list.append(mean_test_loss)
                    if evaluate is not None:
                        mean_test_score /= test_num
                        test_score_list.append(mean_test_score)
                    # change to train model
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
        ax1.set_xlabel("epoch num")
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
            ax2.set_ylim([0, 1.18])
            legend_labels = ["train_loss", "test_loss",
                             "train_score", "test_score"]
            legend_loc = [0.58, 0.77]
        fig.legend(labels=legend_labels, ncol=2, loc=legend_loc)

        if save_fig:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            fig_name = "fig" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            plt.savefig(save_path + fig_name, dpi=200)

        plt.show()


# training
def train_rnn(epoch_num, batch_num, rnn, loss, init_hidden_state, get_params, data_iter, corpus_index,
              num_step, hidden_num, lr, batch_size, char_to_idx, vocab_set, vocab_size, prefixs,
              predict_rnn, pred_num, clipping_theta=1e-2, random_sample=True, device="cuda"):
    """
    function: training and predict in rnn
    params epoch_num: the number of epoch
    params batch_num: the number of batch in a epoch
    params rnn: the rnn model
    params loss: such as nn.CrossEntropyLoss()
    params init_hidden_state: define the state of hidden layer
    params get_params: get the weight and bias in rnn
    params data_iter: data_iter_random/data_iter_consecutive
    params corpus_index: the index of corpus
    params num_step: the number of time step in rnn
    params hidden_num: the number of unit in hidden layer in rnn
    params lr: the learning rate
    params batch_size: the size of a batch
    params char_to_idx: char index which convert Chinese to idx
    params vocab_set: the list of word in corpus
    params vocab_size: the length of vocab_set
    params prefixs: the list include input when you want to predict, such as ["分开", "不分开"]
    params pred_num: the number you want to predict
    params clipping_heta: the max value of the norm of grad
    params random_sample: if sample in random, use data_iter_random. otherwise, use data_iter_consecutive
    params device: "cpu"/"cuda"
    """
    # training bar
    process_bar = sqdm()
    # init
    l_sum, n_class = 0, 0
    # get params in rnn
    params = get_params(vocab_size, hidden_num, vocab_size, device)

    for epoch in range(epoch_num):
        # sample in consecutive
        if not random_sample:
            h_state = init_hidden_state(batch_size, hidden_num, device)
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        for x, y in data_iter(corpus_index, batch_size, num_step, device):
            # x shape: (num_step, batch_size, vocab_size)
            x = to_onehot(x, vocab_size, device)
            # if sample with random, init h_state in each batch
            if random_sample:
                h_state = init_hidden_state(x.shape[1], hidden_num, device)
            else:
                if h_state is not None:
                    if isinstance(h_state, tuple):
                        h_state = (h_state[0].detach_(), h_state[1].detach_())
                    else:
                        # split h_state from cal graph, when sample_consecusive
                        h_state.detach_()

            # rnn, the shape of outputs is (num_step, batch_size, vocab_size)
            outputs, h_state = rnn(x, h_state, params)
            # In order to calculate loss, change outputs shape and y shape
            outputs = outputs.view(-1, outputs.shape[-1])
            y = y.transpose(0, 1).contiguous().view(-1)
            # calculate loss, y --> long type
            l = loss(outputs, y.long())

            # update params
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            # backward
            l.backward()
            # grad clip
            grad_clipping(params, clipping_theta, device)
            # sgd
            sgd(params, lr)

            # loss_sum
            l_sum += l.item() * y.shape[0]
            n_class += y.shape[0]

            # perplexity
            try:
                perplexity = np.exp(l_sum / n_class)
            except OverflowError:
                perplexity = float("inf")

            # training bar
            process_bar.show_process(batch_num, 1, train_loss=perplexity)

        # predict
        print("\n")
        for prefix in prefixs:
            print(f"prefix-{prefix}: ", predict_rnn(prefix, pred_num, rnn, init_hidden_state, hidden_num,
                                                    params, char_to_idx, vocab_set, vocab_size, device))
        print("\n")


# training
def train_rnn_pytorch(epoch_num, batch_num, model, loss, optimizer, data_iter, corpus_index, num_step,
                      batch_size, char_to_idx, vocab_set, vocab_size, prefixs, pred_num, predict_rnn_pytorch,
                      clipping_theta=1e-2, random_sample=True, device="cuda"):
    """
    function: training and predict in rnn
    params epoch_num: the number of epoch
    params batch_num: the number of batch in a epoch
    params model: the rnn model
    params loss: such as nn.CrossEntropyLoss()
    params optimizer: optimizer in pytorch
    params data_iter: data_iter_random/data_iter_consecutive
    params corpus_index: the index of corpus
    params num_step: the number of time step in rnn
    params batch_size: the size of a batch
    params char_to_idx: char index which convert Chinese to idx
    params vocab_set: the list of word in corpus
    params vocab_size: the length of vocab_set
    params prefixs: the list include input when you want to predict, such as ["分开", "不分开"]
    params pred_num: the number you want to predict
    params clipping_heta: the max value of the norm of grad
    params random_sample: if sample in random, use data_iter_random. otherwise, use data_iter_consecutive
    params device: "cpu"/"cuda"
    """
    # training bar
    process_bar = sqdm()
    # init(use in calculate perplexity)
    l_sum, n_class = 0, 0

    for epoch in range(epoch_num):
        # sample in consecutive
        if not random_sample:
            h_state = None
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        for x, y in data_iter(corpus_index, batch_size, num_step, device):
            x = to_onehot(x, vocab_size, device)
            # if sample with random, init h_state in each batch
            if random_sample:
                h_state = None
            else:
                # split h_state from cal graph, when sample_consecutive
                if h_state is not None:
                    if isinstance(h_state, tuple):  # lstm, state: (h, c)
                        h_state = (h_state[0].detach(), h_state[1].detach())
                    else:
                        h_state.detach_()

            # rnn, the shape of outputs is (num_step, batch_size, vocab_size)
            outputs, h_state = model(x, h_state)
            # In order to calculate loss, change outputs shape and y shape
            outputs = outputs.view(-1, outputs.shape[-1])
            y = y.transpose(0, 1).contiguous().view(-1)
            # calculate loss, y --> long type
            l = loss(outputs, y.long())

            # clear grad
            optimizer.zero_grad()
            # grad backward
            l.backward()
            # grad clipping
            grad_clipping(model.parameters(), clipping_theta, device)
            # update grad
            optimizer.step()

            # loss_sum
            l_sum += l.item() * y.shape[0]
            n_class += y.shape[0]

            # calculate perplexity
            try:
                perplexity = np.exp(l_sum / n_class)
            except OverflowError:
                perplexity = float('inf')

            # training bar
            process_bar.show_process(batch_num, 1, train_loss=perplexity)

        # predict
        print("\n")
        for prefix in prefixs:
            print(f"prefix-{prefix}: ", predict_rnn_pytorch(prefix, pred_num, model, char_to_idx,
                                                            vocab_set, vocab_size, device))
        print("\n")
