# -*- coding: utf-8 -*-

from parser.modules.dropout import SharedDropout

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden,
                 activation=nn.LeakyReLU(0.1),
                 dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = activation
        self.dropout = SharedDropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
