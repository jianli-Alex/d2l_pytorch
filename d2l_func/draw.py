#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def set_fig_display_format(pic_format="svg"):
    """设置图片格式：可以是矢量图svg（清晰一点）/也可以其他格式jpge/png/retina/pdf"""
    display.set_matplotlib_formats(pic_format)


def set_fig_display(figsize=(10, 5), font_size=12, pic_format="svg", facecolor="WhiteSmoke",
                    axes_spines_state=[True, True, False, False]):
    """
    功能：设置图片的展示格式 
    参数 figsize：设置图形尺寸（默认(14, 7)）
    参数 font_size：设置字体大小（默认14）
    参数 pic_format：设置图片格式（默认为svg）
    参数 facecolor：设置图片的背景色（默认为白烟色---WhiteSmoke）
    参数 axes_spines_state：设置图片边界的状态（顺序是下左上右，默认下左显示边界，上右不显示）
    """
    # 设置图片格式
    set_fig_display_format()

    # 设置图形尺寸和字体
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.bottom"] = axes_spines_state[0]
    plt.rcParams["axes.spines.left"] = axes_spines_state[1]
    plt.rcParams["axes.spines.top"] = axes_spines_state[2]
    plt.rcParams["axes.spines.right"] = axes_spines_state[3]

    # 设置图片的背景色
    plt.rcParams["axes.facecolor"] = facecolor


def data_iter(batch_size, X, y):
    """
    function: cut X and y in mini-batch data depend on batch_size
    params batch_size: mini-batch GD size
    params X: x in training set
    params y: label in train set
    """
    data_num = len(y)
    index = np.arange(data_num)
    np.random.shuffle(index)

    for i in range(0, data_num, batch_size):
        up_index = np.minimum(i + batch_size, data_num)
        yield X[i:up_index], y[i:up_index]


def bootstrap(x, y):
    """
    function: realize bootstrap by numpy
    """
    data_num = len(y)
    # extract train dataset index and test dataset index
    batch_index = np.random.choice(data_num, size=data_num, replace=True)
    out_index = np.array(list(set(range(data_num)).difference(set(batch_index))))

    # generate train dataset
    xtrain, ytrain = x[batch_index], y[batch_index]
    # generate test dataset
    xtest, ytest = x[out_index], y[out_index]

    return xtrain, ytrain, xtest, ytest
