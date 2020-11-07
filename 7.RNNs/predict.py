#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize the rnn model prediction
"""

import torch
from data_prepare import to_onehot


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


def predict_rnn_pytorch(prefix, pred_num, model, char_to_idx, vocab_set, vocab_size, device):
    """
        function: predict by using rnn network in pytorch
        params prefix: input, such as "分开"
        params pred_num: the number you want to predict
        params model: the rnn model, such as rnn/gru/lstm
        params params: the weight and bias which need to learn
        params char_to_idx: convert chinese to index (char index defined by load_data_jay_song)
        params vocab_set: the list of word in corpus
        params vocab_size: the length of vocab_set
        params device: "cpu"/"cuda"
        """
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
