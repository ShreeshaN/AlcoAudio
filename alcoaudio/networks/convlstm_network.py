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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.dropout0 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=[1, 2])
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=[1, 2])
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=[1, 2])

        self.depth_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.lstm = nn.LSTM(9, 128, 1, bias=True)

        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

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
        x = self.pool3(x)

        x = self.depth_conv(x)

        # LSTM from here
        x = x.squeeze(1).permute(2, 0, 1)  # Converting from (B,C,H,W)->(B,H,W)->(W,B,H)

        lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out.permute(1, 0, 2)  # Converting it back to (B,W,H) from (W,B,H)


        # Flattening to feed it to FFN
        x = lstm_out[-1].view(-1, lstm_out[-1].shape[1])
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
# import torchvision.models as models
# import string
# import numpy as np


# class ConvLSTM(nn.Module):
#     def __init__(self,
#                  backend='resnet18',
#                  rnn_hidden_size=128,
#                  rnn_num_layers=1,
#                  rnn_dropout=0,
#                  seq_proj=[0, 0]):
#         super(ConvLSTM, self).__init__()
#
#         self.num_classes = 1
#
#         self.feature_extractor = getattr(models, backend)(pretrained=True)
#         self.cnn = nn.Sequential(
#                 self.feature_extractor.conv1,
#                 self.feature_extractor.bn1,
#                 self.feature_extractor.relu,
#                 self.feature_extractor.maxpool,
#                 self.feature_extractor.layer1,
#                 self.feature_extractor.layer2,
#                 self.feature_extractor.layer3,
#                 self.feature_extractor.layer4
#         )
#         self.depth_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[2, 11], stride=1)
#
#         self.fully_conv = seq_proj[0] == 0
#         if not self.fully_conv:
#             self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)
#
#         self.rnn_hidden_size = rnn_hidden_size
#         self.rnn_num_layers = rnn_num_layers
#         self.rnn = nn.LSTM(self.get_block_size(self.cnn),
#                           rnn_hidden_size, rnn_num_layers,
#                           batch_first=False,
#                           dropout=rnn_dropout, bidirectional=False)
#         self.linear = nn.Linear(rnn_hidden_size, self.num_classes)
#
#     def forward(self, x, decode=False):
#         x = torch.tensor(x).unsqueeze(1)
#         x = torch.cat((x, x, x), dim=1)
#         hidden = self.init_hidden(x.size(0))
#         features = self.cnn(x)
#         features = self.depth_conv(features)
#         print(features.shape)
#         features = self.features_to_sequence(features)
#         seq, hidden = self.rnn(features, hidden)
#         seq = self.linear(seq)
#         # print("seq", seq.shape)
#         seq = seq.squeeze(0)
#         # print("seq", seq.shape)
#         return seq
#
#     def init_hidden(self, batch_size):
#         h0 = Variable(torch.zeros(self.rnn_num_layers,
#                                   batch_size,
#                                   self.rnn_hidden_size))
#         return h0
#
#     def features_to_sequence(self, features):
#         b, c, h, w = features.size()
#         print(b, c, h, w)
#         assert h == 1, "the height of out must be 1"
#         if not self.fully_conv:
#             features = features.permute(0, 3, 2, 1)
#             features = self.proj(features)
#             features = features.permute(1, 0, 2, 3)
#         else:
#             features = features.permute(3, 0, 2, 1)
#         features = features.squeeze(2)
#         return features
#
#     def get_block_size(self, layer):
#         return layer[-1][-1].bn2.weight.size()[0]
