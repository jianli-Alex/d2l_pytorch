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
        self.bar_length = 70
        self.iter_num = 0

    def show_process(self, data_num, batch_size, mse):
        # update iter_num
        self.iter_num = np.minimum(self.iter_num+batch_size, data_num)

        # the progress of training
        percent = int(self.iter_num/data_num*100)
        num_arrow = int(percent / 100 * self.bar_length)
        num_dash = self.bar_length - num_arrow

        # show training bar
        epoch_bar = f"{self.iter_num}/{data_num} " + "[" + ">" * num_arrow + \
                    "-" * num_dash + "]" + " - " + "loss: " + str(mse) + "\r"

        # stdout write and flush
        sys.stdout.write(epoch_bar)
        sys.stdout.flush()

        # clear zero when a epoch end
        if self.iter_num == data_num:
            self.iter_num = 0
