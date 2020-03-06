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


class ConvLSTM(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.dropout0 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=1)

        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.lstm = nn.LSTM(67, 128, 1, bias=True)

        self.fc1 = nn.Linear(13 * 128, 512)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x = self.reshape_for_pytorch(x)
        x = tensor(x).unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout0(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = self.depth_conv(x)

        # LSTM from here
        x = x.squeeze(1).permute(1, 0, 2)  # Converting from (B,C,W,H)->(B,W,H)->(W,B,H)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)  # Converting it back to (B,W,H) from (W,B,H)

        # Flattening to feed it to FFN
        x = lstm_out.reshape(-1, lstm_out.shape[1:].numel())
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
