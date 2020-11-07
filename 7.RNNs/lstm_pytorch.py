#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize lstm by pytorch without nn.Module
"""

import sys
import torch
import torch.nn as nn
from functools import reduce
sys.path.append("../d2l_func/")
from data_prepare import load_data_jay_song, data_iter_random, data_iter_consecutive, to_onehot
from model_train import train_rnn
from predict import predict_rnn


def get_params(input_num, hidden_num, output_num, device):
    def _ones(shape):
        weight = nn.Parameter(torch.normal(0, 0.01, size=shape, device=device), requires_grad=True)
        return weight

    def _zeros(shape):
        bias = nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
        return bias

    def _three():
        return (
            _ones((input_num, hidden_num)),
            _ones((hidden_num, hidden_num)),
            _zeros(hidden_num)
        )

    # 输入门/遗忘门/输出门
    w_xi, w_hi, b_i = _three()
    w_xf, w_hf, b_f = _three()
    w_xo, w_ho, b_o = _three()
    # 元胞状态
    w_xc, w_hc, b_c = _three()
    # 输出层
    w_hy = _ones((hidden_num, output_num))
    b_y = _zeros(output_num)

    return nn.ParameterList([w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hy, b_y])


def init_hidden_state(batch_size, hidden_num, device):
    return (torch.zeros(batch_size, hidden_num, device=device),
            torch.zeros(batch_size, hidden_num, device=device))


def lstm(inputs, h_state, params):
    w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o, w_xc, w_hc, b_c, w_hy, b_y = params
    outputs = []
    h, c = h_state

    # inputs.shape is (num_step, batch_size, vocab_size)
    for x in inputs:
        it = torch.sigmoid(torch.mm(x, w_xi) + torch.mm(h, w_hi) + b_i)
        ft = torch.sigmoid(torch.mm(x, w_xf) + torch.mm(h, w_hf) + b_f)
        ot = torch.sigmoid(torch.mm(x, w_xo) + torch.mm(h, w_ho) + b_o)
        c_candidate = torch.tanh(torch.mm(x, w_xc) + torch.mm(h, w_hc) + b_c)
        c = it * c_candidate + ft * c
        h = ot * torch.tanh(c)
        y = torch.mm(h, w_hy) + b_y
        outputs.append(y.unsqueeze(0))

    return reduce(lambda x, y: torch.cat((x, y)), outputs), (h, c)


if __name__ == "__main__":
    # load data
    corpus_index, char_to_idx, vocab_set, vocab_size = load_data_jay_song()

    super_params = {
        "epoch_num": 10,
        "rnn": lstm,
        "loss": nn.CrossEntropyLoss(),
        "init_hidden_state": init_hidden_state,
        "hidden_num": 256,
        "get_params": get_params,
        "batch_size": 64,
        "num_step": 32,
        "corpus_index": corpus_index,
        "data_iter": data_iter_random,
        "lr": 10,
        "char_to_idx": char_to_idx,
        "vocab_set": vocab_set,
        "vocab_size": vocab_size,
        "predict_rnn": predict_rnn,
        "pred_num": 50,
        "prefixs": ["分开", "不分开"],
        #     "random_sample": False
    }

    super_params["batch_num"] = len(list(data_iter_random(corpus_index, super_params["batch_size"],
                                                          super_params["num_step"], "cpu")))

    train_rnn(**super_params)
