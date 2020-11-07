#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize gru by pytorch without nn.Module
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
            _zeros(hidden_num),
        )

    w_xr, w_hr, b_r = _three()
    w_xz, w_hz, b_z = _three()
    w_xh, w_hh, b_h = _three()
    w_hy = _ones((hidden_num, output_num))
    b_y = _zeros(output_num)
    return nn.ParameterList([w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hy, b_y])


def init_hidden_state(batch_size, hidden_num, device):
    return torch.zeros(batch_size, hidden_num, device=device)


def gru(inputs, h_state, params):
    w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h, w_hy, b_y = params
    outputs = []

    # inputs.shape is (num_step, batch_size, vocab_size)
    for x in inputs:
        rt = torch.sigmoid(torch.mm(x, w_xr) + torch.mm(h_state, w_hr) + b_r)
        zt = torch.sigmoid(torch.mm(x, w_xz) + torch.mm(h_state, w_hz) + b_z)
        h_candidate = torch.tanh(torch.mm(x, w_xh) + rt * torch.mm(h_state, w_hh) + b_h)
        h_state = zt * h_state + (1 - zt) * h_candidate
        y = torch.mm(h_state, w_hy) + b_y
        outputs.append(y.unsqueeze(0))

    return reduce(lambda x, y: torch.cat((x, y)), outputs), h_state


if __name__ == "__main__":
    # load data
    corpus_index, char_to_idx, vocab_set, vocab_size = load_data_jay_song()

    super_params = {
        "epoch_num": 10,
        "rnn": gru,
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
