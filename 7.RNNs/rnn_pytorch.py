#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize rnn model with sample random or consecutive by pytorch without nn.Module
"""
import sys
import torch
import torch.nn as nn
from functools import reduce
sys.path.append("../d2l_func/")
from data_prepare import load_data_jay_song, data_iter_random, data_iter_consecutive
from data_prepare import onehot, to_onehot
from model_train import train_rnn


# defind rnn params
def get_params(input_num, hidden_num, output_num, device):
    def _ones(shape):
        weight = nn.Parameter(torch.normal(0, 0.01, size=shape, device=device), requires_grad=True)
        return weight

    def _zeros(shape):
        bias = nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
        return bias

    # hidden params
    w_xh = _ones((input_num, hidden_num))
    w_hh = _ones((hidden_num, hidden_num))
    w_hy = _ones((hidden_num, output_num))

    # output params
    b_h = _zeros(hidden_num)
    b_y = _zeros(output_num)
    return nn.ParameterList([w_xh, w_hh, b_h, w_hy, b_y])


# define hidden state
def init_hidden_state(batch_size, hidden_num, device):
    return torch.zeros(batch_size, hidden_num, device=device)


# define rnn layer
def rnn(inputs, h_state, params):
    """
    function: define the process of rnn
    params inputs: shape--> (num_step, batch_size, vocab_size), one-hot vector
    params h_state: the state of hidden layer, shape --> (batch_size, hidden_num)
    params params: the params defined in get_params
    """
    # rnn params
    w_xh, w_hh, b_h, w_hy, b_y = params
    # the number of time_step
    num_step = inputs.shape[0]
    outputs = []

    for step in range(num_step):
        h_state = torch.tanh(torch.mm(inputs[step], w_xh) + torch.mm(h_state, w_hh) + b_h)
        y = torch.mm(h_state, w_hy) + b_y
        outputs.append(y.unsqueeze(0))

    # return shape --> (num_step, batch_size, vocab_size)
    return reduce(lambda x, y: torch.cat((x, y)), outputs), h_state


def predict_rnn(prefix, pred_num, rnn, init_hidden_state, hidden_num,
                params, char_to_idx, vocab_set, vocab_size, device):
    """
    function: predict by using rnn network
    params prefix: input, such as "分开"
    params pred_num: the number you want to predict
    params init_hidden_state: define the state of hidden layer
    params hidden_num: the number of hidden unit
    params params: the weight and bias which need to learn
    params char_to_idx: convert chinese to index (char index defined by load_data_jay_song)
    params vocab_set: the list of word in corpus
    params vocab_size: the length of vocab_set
    params device: "cpu"/"cuda"
    """
    # list which store outputs
    outputs = [char_to_idx[prefix[0]]]
    # define hidden state: batch_size=1
    h_state = init_hidden_state(1, hidden_num, device)

    for i in range(len(prefix) + pred_num - 1):
        inputs = to_onehot(torch.tensor(outputs[-1]).view(-1, 1), vocab_size, device)
        y, h_state = rnn(inputs, h_state, params)

        # if the next word is in prefix
        if i + 1 < len(prefix):
            outputs.append(char_to_idx[prefix[i + 1]])
        else:
            # if the next word is not in prefix, find the best word from predict(max probability)
            outputs.append(y.argmax(dim=2).item())
    return "".join(vocab_set[i] for i in outputs)


if __name__ == "__main__":
    # load data
    corpus_index, char_to_idx, vocab_set, vocab_size = load_data_jay_song()

    super_params = {
        "epoch_num": 5,
        "rnn": rnn,
        "loss": nn.CrossEntropyLoss(),
        "init_hidden_state": init_hidden_state,
        "hidden_num": 256,
        "get_params": get_params,
        "batch_size": 64,
        "num_step": 32,
        "corpus_index": corpus_index,
        "data_iter": data_iter_random,
        "lr": 100,
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
