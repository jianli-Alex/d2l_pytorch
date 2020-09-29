#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize linear model by normal equation, rember wb = [b, w].
with the normal equation, we can get wb = inv(x.T@x)@x.T@y.

Beside, according to experience, when the number of feature is smaller than 1000,
we can realize linear model by solving normal equation fastly, but when the
number of feature is large, especially the number of feature is larger than the
number of sample, the hat matrix(inv(x.T@x)@x.T) is nrivative. It's
advisable to use Gve in this situation.
"""

import numpy as np

"""use normal distribution generate x and error item"""
# data shape
input_num = 10000
# weight and bias
true_w = np.array([2, -3.4])
true_b = np.array([4.2])
# x / label / error
x = np.random.normal(loc=0, scale=1, size=(input_num, len(true_w)))
error = np.random.normal(loc=0, scale=0.01, size=input_num)
y = x @ true_w + true_b + error

"""normal equation solve"""
wb = np.hstack((true_b, true_w))
X = np.hstack((np.ones([input_num, 1]), x))
wb_update = np.linalg.inv(X.T@X)@X.T@y
print(wb_update)