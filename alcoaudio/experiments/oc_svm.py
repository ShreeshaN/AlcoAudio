# -*- coding: utf-8 -*-
"""
@created on: 4/18/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import tensor
import tqdm


def norm(data):
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val)
    return data


def uar(y, yhat):
    return recall_score(y, yhat, average='macro')


#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=[1, 2])
#         self.conv1_bn = nn.BatchNorm2d(64)
#
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=[1, 2])
#         self.conv2_bn = nn.BatchNorm2d(128)
#         self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)
#         self.dropout0 = nn.Dropout(p=0.4)
#
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
#         self.conv3_bn = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=[1, 2])
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)
#
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=[1, 2])
#         self.conv5_bn = nn.BatchNorm2d(32)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         # print('x.size() ', x.size())
#         encoder_op1 = F.relu(self.conv1(x))
#         # print('conv 1', encoder_op1.size())
#         encoder_op2 = F.relu(self.conv2(encoder_op1))
#         # print('conv 2', encoder_op2.size())
#         encoder_op2_pool, pool1_indices = self.pool1(encoder_op2)
#         # print('pool1', encoder_op2_pool.size())
#         encoder_op2_pool = self.dropout0(encoder_op2_pool)
#
#         encoder_op3 = F.relu(self.conv3(encoder_op2_pool))
#         # print('conv 3', encoder_op3.size())
#         encoder_op4 = F.relu(self.conv4(encoder_op3))
#         # print('conv 4', encoder_op4.size())
#         encoder_op4_pool, pool2_indices = self.pool2(encoder_op4)
#         # print('pool2 ', encoder_op4_pool.size(), pool2_indices.shape)
#
#         encoder_op5 = F.relu(self.conv5(encoder_op4_pool))
#         # print('after conv net 5 ', encoder_op5.size())
#
#         return encoder_op5, pool1_indices, pool2_indices
#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#         self.decoder1 = nn.ConvTranspose2d(in_channels=32, out_channels=128, kernel_size=3, stride=[1, 2],
#                                            output_padding=[0, 1])
#         self.decoder1_bn = nn.BatchNorm2d(128)
#         self.unpool1 = nn.MaxUnpool2d(4, stride=1)
#         self.decoder2 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=[1, 2])
#         self.decoder2_bn = nn.BatchNorm2d(256)
#         self.decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=[2, 2],
#                                            output_padding=[0, 1])
#         self.decoder3_bn = nn.BatchNorm2d(128)
#         self.unpool2 = nn.MaxUnpool2d(4, stride=1)
#         self.decoder4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=[1, 2],
#                                            output_padding=[0, 1])
#         self.decoder4_bn = nn.BatchNorm2d(64)
#         self.decoder5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=[1, 2])
#         self.decoder5_bn = nn.BatchNorm2d(1)
#
#     def forward(self, x, pool1_indices, pool2_indices, final_op_shape):
#         # decoder
#         decoder_op1 = F.relu(self.decoder1_bn(self.decoder1(x)))  # , output_size=encoder_op4_pool.size()
#         # print('decoder1', decoder_op1.size())
#         decoder_op1_unpool1 = self.unpool1(decoder_op1, indices=pool2_indices)
#         # print("decoder_op1_unpool1", decoder_op1_unpool1.size())
#         decoder_op2 = F.relu(self.decoder2_bn(self.decoder2(decoder_op1_unpool1)))  # , output_size=encoder_op3.size()
#         # print('decoder2', decoder_op2.size())
#         decoder_op3 = F.relu(self.decoder3_bn(self.decoder3(decoder_op2)))
#         # print('decoder3', decoder_op3.size())
#         decoder_op3_unpool2 = self.unpool2(decoder_op3, indices=pool1_indices)
#         # print("decoder_op3_unpool2", decoder_op3_unpool2.size())
#
#         decoder_op4 = F.relu(self.decoder4_bn(self.decoder4(decoder_op3_unpool2)))
#         # print('decoder4', decoder_op4.size())
#         reconstructed_x = self.decoder5_bn(self.decoder5(decoder_op4, output_size=final_op_shape))
#         # print('decoder5', reconstructed_x.size())
#         return reconstructed_x
#
#
# class ConvAutoEncoder(nn.Module):
#
#     def __init__(self):
#         super(ConvAutoEncoder, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     def forward(self, x):
#         latent_filter_maps, pool1_indices, pool2_indices = self.encoder(x)
#         reconstructed_x = self.decoder(latent_filter_maps, pool1_indices=pool1_indices, pool2_indices=pool2_indices,
#                                        final_op_shape=x.size())
#         return reconstructed_x
#
#
# def get_predcitions(data):
#     batched_input = [data[pos:pos + batch_size] for pos in
#                      range(0, len(data), batch_size)]
#     train_ae_op = []
#     for batch in tqdm.tqdm(batched_input):
#         batch = tensor(batch)
#         pred, _, _ = network.encoder(batch)
#         pred = pred.view(-1, pred.size()[1:].numel()).detach().numpy()
#         train_ae_op.extend(pred)
#     return tensor(train_ae_op)
#
#
# batch_size = 32
# is_cuda_available = torch.cuda.is_available()
# device = torch.device("cuda" if is_cuda_available else "cpu")
# network = ConvAutoEncoder().to(device)
# restore_path = "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/emotion_alco_audio_challenge_data_cae_bo" \
#                "th_classes_1587421865/alco_trained_models/emotion_alco_audio_challenge_data_cae_both_classes_1587421865_27.pt"
# network.load_state_dict(torch.load(restore_path, map_location=device))
# read = 10000000
# train_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_data.npy")[:read]
# train_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_labels.npy")[
#                :read]
# train_data = norm(train_data)
#
# dev_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_data.npy")[:read]
# dev_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_labels.npy")[:read]
# dev_data = norm(dev_data)
#
# test_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_data.npy")[:read]
# test_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy")[:read]
# test_data = norm(test_data)
#
#
# print('Running CAE on train')
# train_data = get_predcitions(tensor(train_data))
# print('Running CAE on dev')
# dev_data = get_predcitions(tensor(dev_data))
# print('Running CAE on test')
# test_data = get_predcitions(tensor(test_data))

train_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_cae_data.npy")
train_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_labels.npy")
dev_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_cae_data.npy")
dev_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_labels.npy")
test_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_cae_data.npy")
test_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy")

inliers_ids, outliers_ids = [i for i, x in enumerate(train_labels) if x == 0], [i for i, x in enumerate(train_labels) if
                                                                                x == 1]
train_inliers, train_outliers = train_data[inliers_ids], train_data[outliers_ids]
# outlier_prop = len(train_outliers) / len(train_inliers)
svm = OneClassSVM(kernel='rbf', nu=0.2, gamma='auto', cache_size=7000)
svm.fit(train_inliers)
dev_y_pred = svm.predict(dev_data)
test_y_pred = svm.predict(test_data)

dev_y_pred = [max(0, x) for x in -dev_y_pred]
print(uar(dev_labels, dev_y_pred))

test_y_pred = [max(0, x) for x in -test_y_pred]
print(uar(test_labels, test_y_pred))
