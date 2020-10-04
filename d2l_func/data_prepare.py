#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

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


def download_data_fashion_mnist(download_path="../data"):
    """
    function: download fashion mnist dataset
    params data_path: download path needed to define. Notably, you don't
                    create the download path even it not exist. when
                    the dataset exist in the download path, it will not
                    download again.
    """
    mnist_train = torchvision.datasets.FashionMNIST(root=download_path, train=True,
                                                    transform=transforms.ToTensor(),
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=download_path, train=False,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
    return mnist_train, mnist_test


def load_data_fashion_mnist(batch_size, num_workers=4, download_path="../data"):
    """use DataLoader to load fashion mnist"""
    # load fashion mnist dataset
    mnist_train, mnist_test = download_data_fashion_mnist(download_path)

    # set the number of process
    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = num_workers

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size,
                                num_workers=num_workers, shuffle=True)
    return train_iter, test_iter