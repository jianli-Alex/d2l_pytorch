#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
unction: generate data for linear model which has 10000 sample, draw a figure with
        the relationship between x and y by sample 1000 data. Finally, use nmpy to
        train the linear model by mini-batch gradient descent algorithm.
params: w = [2, -3.4]
params: b = [4.2]
"""
import sys
sys.path.append("../d2l_func/")
from utils import *
import numpy as np
from twin_fig import draw_twin_fig
from linear_model import LinearModel
from train import *
import warnings
warnings.filterwarnings("ignore")

"""use normal distribution generate x and error item"""
# data shape
input_num = 10000
# weight and bias
w = np.array([2, -3.4])
b = np.array([4.2])
# x / label / error
x = np.random.normal(loc=0, scale=1, size=(input_num, len(w)))
error = np.random.normal(loc=0, scale=0.01, size=input_num)
y = x @ w + b + error

"""draw a figure with the relationship between x and y by sample 1000 data"""
# set figure format
set_fig_display(axes_spines_state=[True, True, True, True])
# sample index
sample_index = np.random.choice(len(y), 1000)
draw_twin_fig(y[sample_index], x[sample_index, 0], x[sample_index, 1])

"""train model by mini-batch GD"""
params = {
    "X": x,
    "y": y,
    "model": LinearModel(),
    "epoch_num": 20,
    "batch_size": 128,
}
model = train(**params)

print(f"w before update is {w}, w after update is {model.w}")
print(f"b before update is {b}, b after update is {model.b}")