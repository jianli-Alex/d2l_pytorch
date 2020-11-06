#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize rnn model by pytorch with nn.Module
"""

import sys
import torch
import torch.nn as nn
sys.path.append("../d2l_func/")
from data_prepare import load_data_jay_song, data_iter_random, data_iter_consecutive
from data_prepare import onehot, to_onehot
from model_train import train_rnn_pytorch


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_num = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)
        self.fc = nn.Linear(self.hidden_num, vocab_size)
        self.h_state = None

    def forward(self, x, h_state):
        # x.shape is (num_step, batch_size, vocab_size)
        # Y.shape is (num_step, batch_size, hidden_num), self.state is (1, batch_size, hidden_num)
        y, self.h_state = self.rnn(x, h_state)
        return self.fc(y), self.h_state


def predict_rnn_pytorch(prefix, pred_num, model, char_to_idx, vocab_set, vocab_size, device):
    outputs = [char_to_idx[prefix[0]]]
    h_state = None

    for i in range(len(prefix) + pred_num - 1):
        # inputs.shape is (batch_size, num_step)
        inputs = to_onehot(torch.tensor(outputs[-1]).view(-1, 1), vocab_size, device)
        if h_state is not None:
            if isinstance(h_state, tuple):
                h_state = (h_state[0].to(device), h_state[1].to(device))
            else:
                h_state.to(device)
        # model inputs is (batch_size, num_step), y.shape is (num_step, batch_size, vocab_size)
        y, h_state = model(inputs, h_state)

        if i + 1 < len(prefix):
            outputs.append(char_to_idx[prefix[i + 1]])
        else:
            outputs.append(y.argmax(dim=2).item())

    return "".join(vocab_set[i] for i in outputs)


if __name__ == "__main__":
    # load data
    corpus_index, char_to_idx, vocab_set, vocab_size = load_data_jay_song()

    # model
    hidden_num = 256
    rnn_layer = nn.RNN(vocab_size, hidden_num)
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
