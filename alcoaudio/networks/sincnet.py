# -*- coding: utf-8 -*-
"""
@created on: 7/6/20,
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


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
                self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincNet(nn.Module):

    def __init__(self, options):
        super(SincNet, self).__init__()
        # self.saved_model = \
        #     torch.load(options['sincnet_saved_model'], map_location='gpu' if torch.cuda.is_available() else 'cpu')[
        #         'CNN_model_par']
        # print(self.saved_model.keys())
        # exit()
        self.batch_size = options['batch_size']
        self.cnn_N_filt = options['cnn_N_filt']
        self.cnn_len_filt = options['cnn_len_filt']
        self.cnn_max_pool_len = options['cnn_max_pool_len']

        self.cnn_act = options['cnn_act']
        self.cnn_drop = options['cnn_drop']

        self.cnn_use_laynorm = options['cnn_use_laynorm']
        self.cnn_use_batchnorm = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']

        self.input_dim = options['input_dim']

        self.fs = options['sampling_rate']

        self.N_cnn_lay = len(options['cnn_N_filt'])
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
            # self.ln0.beta = nn.Parameter(self.saved_model['ln0.beta'])
            # self.ln0.gamma = nn.Parameter(self.saved_model['ln0.gamma'])

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)
            # self.bn0.weight = nn.Parameter(self.saved_model['bn0.weight'])
            # self.bn0.bias = nn.Parameter(self.saved_model['bn0.bias'])
            # self.bn0.running_mean = nn.Parameter(self.saved_model['bn0.running_mean'])
            # self.bn0.running_var = nn.Parameter(self.saved_model['bn0.running_var'])
            # self.bn0.num_batches_tracked = nn.Parameter(self.saved_model['bn0.num_batches_tracked'])

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            ln = LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])])
            # ln.beta = self.saved_model['ln' + str(i) + '.beta']
            # ln.gamma = self.saved_model['ln' + str(i) + '.gamma']
            self.ln.append(ln)

            bn = nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]),
                                momentum=0.05)
            # bn.weight = nn.Parameter(self.saved_model['bn' + str(i) + '.weight'])
            # bn.bias = nn.Parameter(self.saved_model['bn' + str(i) + '.bias'])
            # bn.running_mean = nn.Parameter(self.saved_model['bn' + str(i) + '.running_mean'])
            # bn.running_var = nn.Parameter(self.saved_model['bn' + str(i) + '.running_var'])
            # bn.num_batches_tracked = nn.Parameter(self.saved_model['bn' + str(i) + '.num_batches_tracked'])
            self.bn.append(bn)

            if i == 0:
                self.conv.append(SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

        # self.conv1 = nn.Conv1d(1, 40, 4, 3)
        # self.bn1 = nn.BatchNorm1d(40)
        # self.pool1 = nn.MaxPool1d(2, 2)
        # self.conv2 = nn.Conv1d(40, 40, 4, 3)
        # self.bn2 = nn.BatchNorm1d(40)
        # self.pool2 = nn.MaxPool1d(2, 2)

        self.conv1 = nn.Conv2d(1, 30, (2,3), 2)
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d((1,2), 2)
        self.conv2 = nn.Conv2d(30, 30, (2,3), 2)
        self.bn2 = nn.BatchNorm2d(30)
        self.pool2 = nn.MaxPool2d((1,2), 1 )


        self.fc1 = nn.Linear(55650, 4096)
        self.ln1 = nn.LayerNorm(4096)
        self.drp1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 512)
        self.ln2 = nn.LayerNorm(512)
        self.drp2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, sample):
        output = None
        sample = sample.view(sample.shape[0], 20, self.input_dim)
        for e in range(sample.shape[1]):
            x = sample[:, e, :]
            batch = x.shape[0]
            seq_len = x.shape[1]

            if bool(self.cnn_use_laynorm_inp):
                x = self.ln0(x)

            if bool(self.cnn_use_batchnorm_inp):
                x = self.bn0(x)

            x = x.view(batch, 1, seq_len)
            for i in range(self.N_cnn_lay):

                if self.cnn_use_laynorm[i]:
                    if i == 0:
                        x = self.drop[i](
                                self.act[i](
                                        self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))
                    else:
                        x = self.drop[i](
                                self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

                if self.cnn_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

                if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                    x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
            # xdim = x.shape[1]
            # x = x.view(batch, xdim -1)
            if e == 0:
                output = x
            else:
                output = torch.cat((output, x), dim=2)
        output = self.pool1(self.bn1(F.relu(self.conv1(output.unsqueeze(1)))))
        output = self.pool2(self.bn2(F.relu(self.conv2(output))))
        output = output.view(batch, -1)
        output = F.relu(self.ln1(self.fc1(output)))
        output = self.drp1(output)
        output = F.relu(self.ln1(self.fc2(output)))
        output = self.drp2(output)
        output = self.fc3(output)
        return output
