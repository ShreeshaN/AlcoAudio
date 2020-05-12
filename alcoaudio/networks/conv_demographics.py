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


class ConvNet_Parallel_FFN(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvNet_Parallel_FFN, self).__init__()

        #Conv Layers -------------------------------------------------------------------------

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.dropout0 = nn.Dropout(p=0.4)

        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        # self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1, 2])
        # torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=[1,2])
        # torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=[1,2])
        
        #Parallel FFN-------------------------------------------------------------------------

        self.ff1 = nn.Linear(1, 1)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # self.dropout_f = nn.Dropout(p=0.4)
        # self.ff2 = nn.Linear(24, 5)        

        #FC Layers-------------------------------------------------------------------------        

        self.fc1 = nn.Linear(2561, 256)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 32)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(32, 1)
        # torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x, y):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x = self.reshape_for_pytorch(x)
        # x = x.permute(0, 3, 1, 2)
        x = tensor(x).unsqueeze(1)
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

        # y = F.relu(self.ff1(y))
        # y = self.dropout_f(y)
        # y = F.relu(self.ff2(y))
        # x = torch.cat((x,y),0)

        # Flattening to feed it to FFN
        x = x.view(-1, x.shape[1:].numel())

        x = torch.cat((x,y),1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
