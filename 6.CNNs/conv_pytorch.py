#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize convolution kernel with some situation, such as:
- no padding and stride
- with padding and no stride
- with padding and stride

we only consider the equal width.
"""
import torch
import torch.nn as nn
import numpy as np


class Conv2D(nn.Module):
    """
    function: define convolution layer
    params kernel_size: the shape of convolution kernel (tuple)
    """

    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias


def conv2d(x, k):
    """
    function: realize conv2d kernel calculate
    params x: input data with 2 dimensions (tensor)
    params k: convolution kernel with 2 dimensions (tensor)
    """
    # get convolution kernel shape
    h, w = k.shape
    # define output shape
    output = torch.zeros(x.shape[0] - h + 1, x.shape[1] - w + 1)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = (x[i:(i + h), j:(j + w)] * k).sum()

    return output


def conv2d_padding(x, k, padding=0):
    """
    function: realize conv calculate with padding and no stride
    params x: input data with 2 dimensions (tensor)
    params k: convolution kernel with 2 dimensions (tensor)
    params padding: tuple or int, padding x in equal width, if padding=1, it means
    that padding 0 to x in around, if padding=(2, 3), it means padding two rows 0
    to x in right and left, padding two columns 0 to x in top and bottom
    """
    assert padding >= 0
    if padding == 0:
        return conv2d(x, k)
    else:
        if isinstance(padding, int):
            ph = pw = padding
        else:
            ph, pw = padding

        # padding x
        x = torch.from_numpy(np.pad(x.numpy(), ((ph, ph), (pw, pw))))
        return conv2d(x, k)


def conv2d_padding_stride(x, k, padding=0, stride=1):
    """
    function: realize convolution with padding and stride
    params x: input data with 2 dimensions (tensor)
    params k: convolution kernel with 2 dimensions (tensor)
    params padding: tuple or int, padding x in equal width, if padding=1, it means
    that padding 0 to x in around, if padding=(2, 3), it means padding two rows 0
    to x in right and left, padding two columns 0 to x in top and bottom

    params stride: the stride in conv calculate
    """
    if stride == 1:
        return conv2d_padding(x, k, padding)
    else:
        kh, kw = k.shape
        if isinstance(padding, int):
            ph = pw = padding
        else:
            ph, pw = padding

        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

        # output shape
        output = torch.zeros((x.shape[0] - kh + 2 * ph + sh) // sh,
                             (x.shape[1] - kw + 2 * pw + sw) // sw)
        # padding x
        x = torch.from_numpy(np.pad(x.numpy(), ((ph, ph), (pw, pw))))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = (x[(i * sh):(i * sh + kh), (j * sw):(j * sw + kw)] * k).sum()
        return output


def compute_conv2d(x, conv):
    """
    function: realize convolution by nn.Conv2d
    params x: input data with 2 dimensions (tensor)
    params conv: nn.Conv2d
    """
    # change x from 2 dimension to 4 dimensions, because the convolution kernel
    # in nn.Conv2d is 4 dimensions.
    x = x.view((1, 1) + x.shape)
    result = conv(x)
    return result.view(result.shape[2:])


def conv2d_multi_in(x, k, padding=0, stride=1):
    """
    function: realize conv calculate with multiply channels
    params x: input data with multiply channels (3 dimensions tensor)
    params k: convolution kernel with same channels number in x (3 dimensions tensor)
    params padding and stride: like above
    """
    result = conv2d_padding_stride(x[0], k[0], padding=padding, stride=stride)
    for i in range(1, k.shape[0]):
        result += conv2d_padding_stride(x[i], k[i], padding=padding, stride=stride)
    return result


def conv2d_multi_in_out(X, K, padding=0, stride=1):
    """
    function: realize convolution calculate with multiply input and output channels
    params K: convolution kernel with multiply output channels (4 dimensions tensor)
    """
    result = [conv2d_multi_in(X, k, padding, stride) for k in K]
    return torch.stack(result)


def conv2d_multi_in_out_1x1(x, k):
    """
    function：realize 1x1 convolution
    params x: input data with multiply channels (3 dimension)
    params K: convolution kernel with multiply output channels (4 dimensions tensor)
    """
    # the element in high and width can be view as input data in fc layer
    # the channels can be view as feature in fc layer
    # so change x in the shape of fc layer
    x = x.view(3, -1)
    # stack two channels x
    x = torch.stack((x, x))
    # in order to bmm, change kernel in proper shape
    k = k.view(2, 1, 3)
    result = torch.bmm(k, x)
    return result.view(2, 3, 3)


def pool2d(x, kernel_size, stride=None, mode="max"):
    """
    function: realize max and mean pool
    params x: input data(2 dimension)
    params kernel_size: integer or tuple, like pytorch, the stride shape is
                        same as kernel_size
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if stride == None:
        sh, sw = kh, kw
    else:
        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

    y = torch.zeros(x.shape[0] // sh, x.shape[1] // sw)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode.lower() == "max":
                y[i, j] = x[(i * sh):(i * sh + kh), (j * sw):(j * sw + kw)].max()
            else:
                y[i, j] = x[(i * sh):(i * sh + kh), (j * sw):(j * sw + kw)].mean()

    return y


def pool2d_padding_stride(x, kernel_size, padding=0, stride=None, mode="max"):
    """
    function: realize pool with padding and stride
    """
    if padding == 0:
        return pool2d(x, kernel_size, stride, mode)
    else:
        if isinstance(padding, int):
            ph = pw = padding
        else:
            ph, pw = padding

    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if stride == None:
        sh, sw = kh, kw
    else:
        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

    y = torch.zeros((x.shape[0] - kh + 2 * ph + sh) // sh,
                    (x.shape[1] - kw + 2 * pw + sw) // sw)
    x = torch.from_numpy(np.pad(x.numpy(), ((ph, ph), (pw, pw))))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode.lower() == "max":
                y[i, j] = x[(i * sh):(i * sh + kh), (j * sw):(j * sw + kw)].max()
            else:
                y[i, j] = x[(i * sh):(i * sh + kh), (j * sw):(j * sw + kw)].mean()
    return y


def pool2d_multi_in_out(x, kernel_size, padding=0, stride=None, mode="max"):
    """
    function: realize pooling with multiply channels
    """
    result = [pool2d_padding_stride(x[k], kernel_size[1:], padding, stride,
                                    mode)for k in range(kernel_size[0])]
    return torch.stack(result)


if __name__ == "__main__":
    # test1: convolution calculation
    x = torch.arange(9).view(3, 3)
    k = torch.arange(4).view(2, 2)
    print("test1: \n", conv2d(x, k), "\n")

    # test2: convolution with padding calculation
    print("test2: ")
    x = torch.arange(9.).view(3, 3)
    k = torch.arange(4.).view(2, 2)
    print("self-defined")
    print(conv2d_padding(x, k, padding=1))
    print("nn.Conv2d")
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2,
                     padding=1, stride=1)
    conv.weight.data = k.view((1, 1) + k.shape)
    conv.bias.data = torch.zeros(1)
    print(compute_conv2d(x, conv), "\n")

    # test3: convolution with padding and stride
    print("test3: ")
    test_x = torch.rand(8, 8)
    test_k = torch.rand(3, 5)
    padding = (0, 1)
    stride = (3, 4)
    print("self-defined")
    print(conv2d_padding_stride(test_x, test_k, padding, stride))
    print(conv2d_padding_stride(test_x, test_k, padding, stride).shape)
    print("nn.Conv2d")
    conv = nn.Conv2d(in_channels=1, out_channels=1,
                     kernel_size=3, padding=padding, stride=stride)
    conv.weight.data = test_k.view((1, 1) + test_k.shape)
    conv.bias.data = torch.zeros(1)
    print(compute_conv2d(test_x, conv), "\n")

    # test4: multiply input channels
    print("test4")
    x1, x2 = torch.arange(9).view(3, 3), torch.arange(1, 10).view(3, 3)
    k1, k2 = torch.arange(4).view(2, 2), torch.arange(1, 5).view(2, 2)
    X, K = torch.stack((x1, x2)), torch.stack((k1, k2))
    print(conv2d_multi_in(X, K), "\n")

    # test5: multiply output channels
    print("test5")
    test_k = torch.stack((K, K + 1, K + 2))
    print(conv2d_multi_in_out(X, test_k), "\n")

    # test6: 1x1 convolution
    print("test6")
    print("1x1 conv")
    x = torch.arange(27).view(3, 3, 3)
    k = torch.arange(6).view(2, 3, 1, 1)
    print(conv2d_multi_in_out_1x1(x, k))
    print("conv_multi_in_out realize")
    print(conv2d_multi_in_out(x, k), "\n")

    # test7: pooling with stride and no padding
    print("test7: ")
    x = torch.arange(9.).view(3, 3)
    print("self-define")
    print(pool2d(x, (2, 2)))
    print(pool2d(x, (2, 2), mode="mean"))
    print("MaxPool2d")
    pool = nn.MaxPool2d(kernel_size=2)
    print(pool(x.view((1, 1) + x.shape)))
    print("AvgPool2d")
    pool = nn.AvgPool2d(kernel_size=2)
    print(pool(x.view((1, 1) + x.shape)), "\n")

    # test 8: pooling with stride and padding
    print("test8: ")
    print("self-define")
    x = torch.arange(16.).view(4, 4)
    print(pool2d_padding_stride(x, kernel_size=3))
    print(pool2d_padding_stride(x, kernel_size=(3, 3), padding=1, stride=2))
    print(pool2d_padding_stride(x, kernel_size=(2, 4),
                                padding=(1, 2), stride=(2, 3), mode="mean"))
    print("MaxPool2d")
    pool = nn.MaxPool2d(kernel_size=3)
    print(pool(x.view((1, 1) + x.shape)))
    pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    print(pool(x.view((1, 1) + x.shape)))
    print("AvgPool2d")
    pool = nn.AvgPool2d(kernel_size=(2, 4), padding=(1, 2), stride=(2, 3))
    print(pool(x.view((1, 1) + x.shape)), "\n")

    # test 9: pooling with multiply channels
    print("test 9")
    print("self-define")
    x = torch.arange(16.).view(4, 4)
    X = torch.stack((x, x + 1, x + 2))
    print(pool2d_multi_in_out(X, (3, 2, 2), padding=1, stride=2))
    print("MaxPool2d")
    pool = nn.MaxPool2d(kernel_size=2, padding=1, stride=2)
    print(pool(X))
