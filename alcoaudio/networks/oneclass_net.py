# -*- coding: utf-8 -*-
"""
@created on: 4/19/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=[1, 2])
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)
        self.dropout0 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=[1, 2])
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=[1, 2])
        self.conv5_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        # print('x.size() ', x.size())
        encoder_op1 = F.relu(self.conv1(x))
        # print('conv 1', encoder_op1.size())
        encoder_op2 = F.relu(self.conv2(encoder_op1))
        # print('conv 2', encoder_op2.size())
        encoder_op2_pool, pool1_indices = self.pool1(encoder_op2)
        # print('pool1', encoder_op2_pool.size())
        encoder_op2_pool = self.dropout0(encoder_op2_pool)

        encoder_op3 = F.relu(self.conv3(encoder_op2_pool))
        # print('conv 3', encoder_op3.size())
        encoder_op4 = F.relu(self.conv4(encoder_op3))
        # print('conv 4', encoder_op4.size())
        encoder_op4_pool, pool2_indices = self.pool2(encoder_op4)
        # print('pool2 ', encoder_op4_pool.size())

        encoder_op5 = F.relu(self.conv5(encoder_op4_pool))
        # print('after conv net 5 ', encoder_op5.size())

        return encoder_op5, pool1_indices, pool2_indices


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder1 = nn.ConvTranspose2d(in_channels=32, out_channels=128, kernel_size=3, stride=[1, 2])
        self.decoder1_bn = nn.BatchNorm2d(128)
        self.unpool1 = nn.MaxUnpool2d(4, stride=1)
        self.decoder2 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=[1, 2])
        self.decoder2_bn = nn.BatchNorm2d(256)
        self.decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=[2, 2])
        self.decoder3_bn = nn.BatchNorm2d(128)
        self.unpool2 = nn.MaxUnpool2d(4, stride=1)
        self.decoder4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=[1, 2])
        self.decoder4_bn = nn.BatchNorm2d(64)
        self.decoder5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=[1, 2])
        self.decoder5_bn = nn.BatchNorm2d(1)

    def forward(self, x, pool1_indices, pool2_indices, final_op_shape):
        # decoder
        decoder_op1 = F.relu(self.decoder1_bn(self.decoder1(x)))  # , output_size=encoder_op4_pool.size()
        # print('decoder1', decoder_op1.size())
        decoder_op1_unpool1 = self.unpool1(decoder_op1, indices=pool2_indices)
        # print("decoder_op1_unpool1", decoder_op1_unpool1.size())
        decoder_op2 = F.relu(self.decoder2_bn(self.decoder2(decoder_op1_unpool1)))  # , output_size=encoder_op3.size()
        # print('decoder2', decoder_op2.size())
        decoder_op3 = F.relu(self.decoder3_bn(self.decoder3(decoder_op2)))
        # print('decoder3', decoder_op3.size())
        decoder_op3_unpool2 = self.unpool2(decoder_op3, indices=pool1_indices)
        # print("decoder_op3_unpool2", decoder_op3_unpool2.size())

        decoder_op4 = F.relu(self.decoder4_bn(self.decoder4(decoder_op3_unpool2)))
        # print('decoder4', decoder_op4.size())
        reconstructed_x = self.decoder5_bn(self.decoder5(decoder_op4, output_size=final_op_shape))
        # print('decoder5', reconstructed_x.size())
        return reconstructed_x


class ConvAutoEncoder(nn.Module):

    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.unsqueeze(1)
        latent_filter_maps, pool1_indices, pool2_indices = self.encoder(x)
        reconstructed_x = self.decoder(latent_filter_maps, pool1_indices=pool1_indices, pool2_indices=pool2_indices,
                                       final_op_shape=x.size())
        return reconstructed_x


class OneClassNN(nn.Module):
    def __init__(self):
        super(OneClassNN, self).__init__()

        self.w = nn.Linear(11520, 2048)
        self.dropout1 = nn.Dropout(p=0.3)
        self.v = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.relu(self.w(x))
        x = self.dropout1(x)
        return self.v(x), self.w, self.v


class OneClassCAE(nn.Module):
    def __init__(self):
        super(OneClassCAE, self).__init__()
        self.encoder = Encoder()
        self.one_class_nn = OneClassNN()

    def forward(self, x):
        latent_filter_maps = self.encoder(x)
        latent_vector = latent_filter_maps.view(-1, x.size()[1:].numel())
        y_hat, w, v = self.one_class_nn(latent_vector)
        return y_hat, w, v
