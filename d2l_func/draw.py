#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from IPython import display


def set_fig_display_format(pic_format="svg"):
    """set figure format, such as svg(default), jpg, png, retina, pdf"""
    display.set_matplotlib_formats(pic_format)


def set_fig_display(figsize=(10, 5), font_size=12, pic_format="svg", facecolor="WhiteSmoke",
                    axes_spines_state=[True, True, False, False]):
    """
    function: set show figure format
    params figsize: set figure size, default(14, 7)
    params font_size: set font size, default 14
    params pic_format: set figure format, default svg
    params facecolor: set figure background color, default "WhiteSmoke"
    params axes_spines_state: set the state of figure border, the order is
                                bottom-left-top-right, default show left and
                                bottom border
    """
    # set figure format
    set_fig_display_format()

    # set figure size and fontsize
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.bottom"] = axes_spines_state[0]
    plt.rcParams["axes.spines.left"] = axes_spines_state[1]
    plt.rcParams["axes.spines.top"] = axes_spines_state[2]
    plt.rcParams["axes.spines.right"] = axes_spines_state[3]

    # set figure background color
    plt.rcParams["axes.facecolor"] = facecolor


def get_fashion_mnist_label(labels):
    """
    function: replace fashion mnist dataset label, original label is number,
    the label change text after replace
    """
    text_labels = ["t-shirt", "trouse", "pullover", "dress", "coat", "sandal",
                   "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(image, labels):
    """function: show fashion minst dataset figure"""
    # set figure format
    set_fig_display()

    _, axes = plt.subplots(1, len(image), figsize=(12, 12))
    for ax, img, label in zip(axes, image, labels):
        # need to change 28*28 when show picture, fashion-mnist is 1*28*28
        ax.imshow(img.view(28, 28))
        ax.set_title(label)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()
