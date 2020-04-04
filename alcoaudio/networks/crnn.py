# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
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
import numpy as np


class ConvModule(nn.Module):
    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.dropout0 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=[1, 2])
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=[1, 2])
        self.conv5_bn = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=[1, 2])

        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout0(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.depth_conv(x)
        return x


class RNNModule(nn.Module):
    def __init__(self):
        super(RNNModule, self).__init__()
        self.lstm = nn.LSTM(1, 128, 2, bias=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(5120, 1)

    def forward(self, x):
        rnn_out, _ = self.lstm(x)

        # (seq_len, batch, num_directions * hidden_size) -> (batch, num_directions * hidden_size, seq_len)
        rnn_out = rnn_out.permute(1, 2, 0)

        x = rnn_out.contiguous().view(-1, rnn_out.shape[1] * rnn_out.shape[2])
        x = self.fc1(x)
        return x


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = ConvModule()
        self.rnn = RNNModule()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = tensor(x).unsqueeze(1)
        x = self.cnn(x)

        # LSTM from here
        x = x.squeeze(1).permute(2, 0, 1)  # Converting from (B,C,H,W)->(B,H,W)->(W,B,H)
        output = self.rnn(x)
        return output
