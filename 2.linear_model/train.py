#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys

sys.path.append("../d2l_func/")
from utils import *
from sqdm import sqdm
import pandas as pd
pd.set_option('display.max_rows',None)#取消行限制
pd.set_option('display.width',1000)#增加每行的宽度

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
