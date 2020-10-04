#! /usr/bin/env python
# -*-coding: utf-8 -*-

import numpy as np


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