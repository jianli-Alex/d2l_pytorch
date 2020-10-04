#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys

sys.path.append("../d2l_func/")
from data_prepare import data_iter
from sqdm import sqdm

process_bar = sqdm()


def train(X, y, model, epoch_num, batch_size):
    for epoch in range(epoch_num):
        print(f"Epoch [{epoch + 1}/{epoch_num}]")
        for xdata, ydata in data_iter(batch_size, X, y):
            model.fit(xdata, ydata)
            mse = model.score(xdata, ydata)
            process_bar.show_process(len(y), batch_size, mse)
        print("\n")

    return model
