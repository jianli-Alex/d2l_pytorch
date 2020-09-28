#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys

sys.path.append("../d2l_func/")
from utils import *
from sqdm import sqdm

process_bar = sqdm()


def train(X, y, model, epoch_num, batch_size):
    for epoch in range(epoch_num):
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        for xdata, ydata in data_iter(batch_size, X, y):
            model.fit(xdata, ydata)
            mse = np.round(model.score(xdata, ydata), 5)
            process_bar.show_process(len(y), batch_size, mse)
        print("\n")

    return model
