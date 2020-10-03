#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import numpy as np

class sqdm(object):
    """
    function: show training process by a training bar
    params data_num: total number of training sample
    params batch_size: mini-batch size
    params mse: Mean square error in each iteration
    """

    def __init__(self):
        self.bar_length = 30
        self.iter_num = 0

    def show_process(self, data_num, batch_size, train_loss="-", train_score="-", test_loss="-", test_score="-"):
        # update iter_num
        self.iter_num = np.minimum(self.iter_num + batch_size, data_num)

        # the progress of training
        percent = int(self.iter_num / data_num * 100)
        num_arrow = int(percent / 100 * self.bar_length)
        num_dash = self.bar_length - num_arrow

        # limit decimal
        if isinstance(train_loss, float):
            train_loss = "%.4f" % (train_loss)
        if isinstance(train_score, float):
            train_score = "%.2f" % (train_score)
        if isinstance(test_loss, float):
            test_loss = "%.4f" % (test_loss)
        if isinstance(test_score, float):
            test_score = "%.2f" % (test_score)

        # show training bar
        epoch_bar = f"{self.iter_num}/{data_num} " + "[" + ">" * num_arrow + \
                    "-" * num_dash + "]" + " - " + "train_loss: " + train_loss + ", " + \
                    "train_score: " + train_score + ", " + "test_loss: " + test_loss + \
                    ", " + "test_score: " + test_score + "\r"

        # stdout write and flush
        sys.stdout.write(epoch_bar)
        sys.stdout.flush()

        # clear zero when a epoch end
        if self.iter_num == data_num:
            self.iter_num = 0
