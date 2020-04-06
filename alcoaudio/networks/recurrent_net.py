# -*- coding: utf-8 -*-
"""
@created on: 4/5/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch import tensor
import cv2


class RNNModule(nn.Module):
    def __init__(self):
        super(RNNModule, self).__init__()
        self.lstm = nn.GRU(40, 128, 2, bias=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(512*690, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = tensor(x).permute(2, 0, 1)  # (B, H, W) -> (W, B, H)
        rnn_out, _ = self.lstm(x)
        rnn_out = rnn_out.permute(1, 2,
                                  0)  # (seq_len, batch, num_directions * hidden_size) -> (batch, num_directions * hidden_size, seq_len)
        x = rnn_out.contiguous().view(-1, rnn_out.shape[1] * rnn_out.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
