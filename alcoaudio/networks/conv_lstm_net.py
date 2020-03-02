# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Pratik J ,
@version: v0.0.1
@system name: pratcr7
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


class LSTMNet(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """


        super().__init__()

        self.conv1 = nn.Conv2d(1, 256, 3, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.pool1 = nn.MaxPool1d(8)

        self.conv3 = nn.Conv2d(256, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.conv6 = nn.Conv2d(64, 64, 3, 1)

        self.lstm1 = nn.LSTM(64, 64, 1, batch_first=True, dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True, dropout=0.3, bidirectional=False)

        self.fc1 = nn.Linear(26 * 64, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = tensor(x).unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = F.relu(self.conv6(x))
        x = x.view(-1, x.shape[1:].numel())
        h_0 = Variable(torch.zeros(1, x.size(0), 64))
        c_0 = Variable(torch.zeros(1, x.size(0), 64))
        ula1, (x, _) = F.relu(self.lstm1(x, (h_0, c_0)))
        ula2, (x, _) = F.relu(self.lstm2(x, (h_0, c_0)))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
