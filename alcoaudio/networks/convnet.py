# -*- coding: utf-8 -*-
"""
@created on: 3/7/20,
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


class ConvNet(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvNet, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        # self.conv1_bn.track_running_stats = False
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        # self.conv2_bn.track_running_stats = False
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.dropout0 = nn.Dropout(p=0.4)

        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        # self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1, 2])
        self.conv4_bn = nn.BatchNorm2d(64)
        # self.conv4_bn.track_running_stats = False
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1, 2])
        self.conv5_bn = nn.BatchNorm2d(64)
        # self.conv5_bn.track_running_stats = False
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=[1, 2])

        self.fc1 = nn.Linear(40 * 64, 256)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x = self.reshape_for_pytorch(x)
        # x = x.permute(0, 3, 1, 2)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout0(x)

        # x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool3(x)
        x = x.unsqueeze(2)

        # Flattening to feed it to FFN
        x = x.view(-1, x.shape[1:].numel())

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
