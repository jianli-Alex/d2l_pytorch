#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
function: realize rnn model
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_num = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)
        self.vocab_size = vocab_size
        self.fc = nn.Linear(self.hidden_num, vocab_size)
        self.h_state = None

    def forward(self, x, h_state):
        # x.shape is (num_step, batch_size, vocab_size), h_state.shape is (batch_size, hidden_num)
        y, self.h_state = self.rnn(x, h_state)
        return self.fc(y), self.h_state
