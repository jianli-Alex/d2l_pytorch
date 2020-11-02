#! /usr/bin/env python
# -*-coding: utf-8 -*-

import sys
import torch
import zipfile
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


def download_data_fashion_mnist(download_path="../data", resize=None):
    """
    function: download fashion mnist dataset
    params data_path: download path needed to define. Notably, you don't
                    create the download path even it not exist. when
                    the dataset exist in the download path, it will not
                    download again.
    """
    trans = []
    # 'Resize' can be used to change the resolution of figure
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    # use Compose to combine 'ToTensor' and 'Resize'
    transform = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=download_path, train=True,
                                                    download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=download_path, train=False,
                                                   download=True, transform=transform)
    return mnist_train, mnist_test


def load_data_fashion_mnist(batch_size, num_workers=4,
                            download_path="../data", resize=None, **kwargs):
    """use DataLoader to load fashion mnist"""
    # load fashion mnist dataset
    mnist_train, mnist_test = download_data_fashion_mnist(download_path, resize)

    # set the number of process
    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = num_workers

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True, **kwargs)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size,
                                num_workers=num_workers, shuffle=True, **kwargs)
    return train_iter, test_iter


def zip_data_jay_song():
    # extract data from jay zipfile
    with zipfile.ZipFile("../data/jaychou_lyrics.txt.zip") as zip_file:
        with zip_file.open("jaychou_lyrics.txt", "r") as f:
            corpus_char = f.read().decode("utf-8")

    # substitute "\n" to space
    return corpus_char.replace("\n", " ").replace("\r", " ")


def load_data_jay_song(sort=True):
    """
    function: function: load the data of jay song which use in RNN
    params corpus_char: the complete char of corpus
    """
    corpus_char = zip_data_jay_song()
    # idx to char
    vocab_set = list(set(corpus_char))
    if sort:
        vocab_set.sort()
    # char to idx
    char_to_idx = {char: i for i, char in enumerate(vocab_set)}
    # vocab size
    vocab_size = len(vocab_set)
    # the index of corpus
    corpus_index = [char_to_idx[i] for i in corpus_char]

    return corpus_index, char_to_idx, vocab_set, vocab_size


def data_iter_random(corpus_index, batch_size, num_steps):
    """
    function: realize random sample
    params corpus_index: the idx with corpus --> list
    params batch_size: the size of each batch
    params num_steps: the number of time steps in a network
    """
    # because the index of y is equal to the index of x + 1, so when we calucate the number of example,
    # we should use len(corpus_index) - 1, "example_num" stand fot the number of example with the num_steps.
    example_num = (len(corpus_index) - 1) // num_steps
    sample_start = (len(corpus_index) - 1) % num_steps
    # the example is the combination of several char, so it must with num_steps. when the example is not the
    # factor of batch_size, we only retain the complete example
    sample_start = np.random.randint(sample_start + 1)
    example_index = np.arange(sample_start, len(corpus_index), num_steps)[:example_num]
    np.random.shuffle(example_index)

    # extract batch example
    for idx in np.arange(0, len(example_index), batch_size):
        batch_example = example_index[idx:(idx+batch_size)]
        # extract example in each batch
        x = [corpus_index[pos:(pos+num_steps)] for pos in batch_example]
        y = [corpus_index[(pos+1):(pos+1+num_steps)] for pos in batch_example]
        yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def data_iter_consecutive(corpus_index, batch_size, num_step):
    """realize consecutive sample, the params is same as random sample"""
    # change to tensor in order to the use "view"
    corpus_index = torch.tensor(corpus_index, dtype=torch.float32)
    # the length of batch if only split in a batch with batch_size
    batch_len = len(corpus_index) // batch_size
    # sample batch
    batch_start = np.random.randint(len(corpus_index) % batch_size + 1)
    corpus_index = corpus_index[batch_start: batch_start + batch_size * batch_len].view(batch_size, -1)

    # sample example
    sample_start = np.random.randint((batch_len - 1) % num_step + 1)
    # the max num generate with batch_size
    batch_num = (batch_len - 1) // num_step

    # yield consecutive sample with num_step
    for i in range(batch_num):
        i = sample_start + i * num_step
        x = corpus_index[:, i:(i + num_step)]
        y = corpus_index[:, (i + 1):(i + 1 + num_step)]
        yield x, y
