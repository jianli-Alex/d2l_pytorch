#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize lstm by pytorch without nn.Module
"""

import sys
import torch
import torch.nn as nn
sys.path.append("../d2l_func/")
from data_prepare import load_data_jay_song, data_iter_random, data_iter_consecutive, to_onehot
from model_train import train_rnn_pytorch
from predict import predict_rnn_pytorch
from rnn_model import RNNModel


if __name__ == "__main__":
    # load data
    corpus_index, char_to_idx, vocab_set, vocab_size = load_data_jay_song()
    # model
    hidden_num = 256
    rnn_layer = nn.LSTM(vocab_size, hidden_num)
    model = RNNModel(rnn_layer, vocab_size)
    model = model.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    params = {
        "epoch_num": 10,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "batch_size": 64,
        "num_step": 32,
        "corpus_index": corpus_index,
        "data_iter": data_iter_consecutive,
        "char_to_idx": char_to_idx,
        "vocab_set": vocab_set,
        "vocab_size": vocab_size,
        "predict_rnn_pytorch": predict_rnn_pytorch,
        "pred_num": 50,
        "prefixs": ["分开", "不分开"],
        "random_sample": False
    }

    params["batch_num"] = len(list(data_iter_consecutive(corpus_index, params["batch_size"],
                                                         params["num_step"], "cpu")))

    train_rnn_pytorch(**params)
