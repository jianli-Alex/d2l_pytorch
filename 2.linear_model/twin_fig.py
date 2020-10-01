#! /usr/bin/env python
# -*-coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def draw_twin_fig(bottom_data, left_data, right_data):
    """
    function: draw a figure with twin axes y
    params bottom_data： data draw in x axes
    params left_data: data draw in left y axes
    params right_data: data draw in  right y axes
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(bottom_data, left_data, "o", label="feature 1")
    ax1.set_xlabel("ydata")
    ax1.set_ylabel("feature 1")
    ax1.legend(loc=[0.8, 0.9])

    ax2 = ax1.twinx()
    ax2.plot(bottom_data, right_data, "ro", label="feature 2", alpha=0.7)
    ax2.set_ylabel("feature 2")
    ax2.legend(loc=[0.8, 0.8])

    plt.title("linear model")
    plt.savefig("./img/linear_model.png", dpi=200)
    plt.show()