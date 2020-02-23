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


class ConvNet(nn.Module):

    def __init__(self, weights):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        def prepare_weights(paraemters):
            weight = tensor(paraemters).permute(dims=(2, 1, 0))
            return weight

        def prepare_bias(parameters):
            return tensor(parameters)

        super().__init__()

        self.conv1 = nn.Conv1d(1, 128, 5, 1)
        weight = prepare_weights(weights[0])
        bias = prepare_bias(weights[1])
        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=True)

        self.conv2 = nn.Conv1d(128, 128, 5, 1)
        weight = prepare_weights(weights[2])
        bias = prepare_bias(weights[3])
        self.conv2.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv2.bias = torch.nn.Parameter(bias, requires_grad=True)

        self.dropout = nn.Dropout(p=0.1)
        self.pool1 = nn.MaxPool1d(8)

        self.conv3 = nn.Conv1d(128, 128, 5, 1)
        weight = prepare_weights(weights[4])
        bias = prepare_bias(weights[5])
        self.conv3.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv3.bias = torch.nn.Parameter(bias, requires_grad=True)

        self.conv4 = nn.Conv1d(128, 128, 5, 1)
        weight = prepare_weights(weights[6])
        bias = prepare_bias(weights[7])
        self.conv4.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv4.bias = torch.nn.Parameter(bias, requires_grad=True)

        self.conv5 = nn.Conv1d(128, 128, 5, 1)
        weight = prepare_weights(weights[8])
        bias = prepare_bias(weights[9])
        self.conv5.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv5.bias = torch.nn.Parameter(bias, requires_grad=True)
        self.dropout = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv1d(128, 128, 5, 1)
        weight = prepare_weights(weights[10])
        bias = prepare_bias(weights[11])
        self.conv6.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.conv6.bias = torch.nn.Parameter(bias, requires_grad=True)
        self.fc1 = nn.Linear(23 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

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
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = F.relu(self.conv6(x))

        x = x.view(-1, x.shape[1:].numel())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
