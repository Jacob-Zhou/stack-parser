# -*- coding: utf-8 -*-

import torch.nn as nn


class Highway(nn.Module):

    def __init__(self, size, n_layers=1, dropout=0.5):
        super(Highway, self).__init__()

        self.size = size
        self.n_layers = n_layers
        self.trans = nn.ModuleList(nn.Linear(size, size)
                                   for _ in range(n_layers))
        self.gates = nn.ModuleList(nn.Linear(size, size)
                                   for _ in range(n_layers))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        g = self.gates[0](x).sigmoid()
        h = self.trans[0](x).relu()
        x = g * h + (1 - g) * x

        for i in range(1, self.n_layers):
            x = self.dropout(x)
            g = self.gates[i](x).sigmoid()
            h = self.trans[i](x).relu()
            x = g * h + (1 - g) * x

        return x
