# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch import tensor
import time
import json

from alcoaudio.networks.conv_network import ConvNet
from alcoaudio.utils import file_utils
from alcoaudio.datagen.audio_feature_extractors import preprocess_data
from alcoaudio.utils.network_utils import accuracy_fn, log_summary
from alcoaudio.utils.data_utils import read_h5py, read_npy


class ConvNetRunner:
    def __init__(self, args):
        self.run_name = args.run_name + '_' + str(time.time()).split('.')[0]
        self.current_run_basepath = args.network_metrics_basepath + '/' + self.run_name + '/'
        self.learning_rate = args.learning_rate
        self.transfer_learning_rate = args.transfer_learning_rate
        self.epochs = args.epochs
        self.test_net = args.test_net
        self.train_net = args.train_net
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.audio_basepath = args.audio_basepath
        self.train_data_file = args.train_data_file
        self.test_data_file = args.test_data_file
        self.data_read_path = args.data_save_path
        self.is_cuda_available = torch.cuda.is_available()
        self.display_interval = args.display_interval
        self.sampling_rate = args.sampling_rate
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.overlap = args.overlap

        self.network_metrics_basepath = args.network_metrics_basepath
        self.tensorboard_summary_path = self.current_run_basepath + args.tensorboard_summary_path
        self.network_save_path = self.current_run_basepath + args.network_save_path

        self.network_restore_path = args.network_restore_path

        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_save_interval = args.network_save_interval
        self.normalise = args.normalise
        self.dropout = args.dropout
        self.threshold = args.threshold

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        # Loading keras model weights
        self.weights = np.load(args.keras_model_weights, allow_pickle=True)

        self.network = None
        self.network = ConvNet(self.weights).to(self.device)

        self.loss_function = nn.BCELoss()

        self.optimiser = optim.Adam(
                [{'params': list(self.network.parameters())[:12], 'lr': self.transfer_learning_rate},
                 {'params': list(self.network.parameters())[12:], 'lr': self.learning_rate}])

        if self.train_net:
            self.network.train()
            self.log_file = open(self.network_save_path + '/' + self.run_name + '.log', 'w')
            self.log_file.write(json.dumps(args))
        if self.test_net:
            print('Loading Network')
            self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
            self.network.eval()
            self.log_file = open(self.network_restore_path + '/' + self.run_name + '.log', 'a')
            print('\n\n\n********************************************************', file=self.log_file)
            print('Testing Model - ', self.network_restore_path)
            print('Testing Model - ', self.network_restore_path, file=self.log_file)
            print('********************************************************', file=self.log_file)

        self.writer = SummaryWriter(self.tensorboard_summary_path)
        print("Network config:\n", self.network)
        print("Network config:\n", self.network, file=self.log_file)

        self.batch_loss, self.batch_accuracy, self.uar = [], [], []

        print('Configs used:\n', json.dumps(args, indent=4))
        print('Configs used:\n', json.dumps(args, indent=4), file=self.log_file)

    def data_reader(self, data_filepath, label_filepath, should_batch=True, shuffle=True):
        # data = pd.read_csv(data_file)[:50]
        # if shuffle:
        #     data = data.sample(frac=1)
        # input_data, labels = preprocess_data(self.audio_basepath, data['WAV_PATH'].values, data['label'].values,
        #                                      normalise=normalise, sample_size_in_seconds=self.sample_size_in_seconds,
        #                                      sampling_rate=self.sampling_rate, overlap=self.overlap)
        input_data, labels = read_npy(data_filepath), read_npy(label_filepath)

        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(input_data), self.batch_size)]
            batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
            return batched_input, batched_labels
        else:
            return input_data, labels

    def train(self):
        train_data, train_labels = self.data_reader(self.data_read_path + 'train_data.npy',
                                                    self.data_read_path + 'train_labels.npy', shuffle=False)
        test_data, test_labels = self.data_reader(self.data_read_path + 'test_data.npy',
                                                  self.data_read_path + 'test_labels.npy', shuffle=False)
        total_step = len(train_data)
        for epoch in range(1, self.epochs):
            self.batch_loss, self.batch_accuracy, self.batch_uar = [], [], []
            for i, (audio_data, label) in enumerate(zip(train_data, train_labels)):
                print("train", audio_data.shape)
                predictions = self.network(audio_data)
                predictions = nn.Sigmoid()(predictions).squeeze(1)
                loss = self.loss_function(predictions, tensor(label).float())
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                accuracy, uar = accuracy_fn(predictions, label, self.threshold)
                self.batch_loss.append(loss.detach().numpy())
                self.batch_accuracy.append(accuracy)
                self.batch_uar.append(uar)
                if i % self.display_interval == 0:
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}")
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}",
                            file=self.log_file)

            # Test data
            self.test_batch_loss, self.test_batch_accuracy, self.test_batch_uar = [], [], []
            with torch.no_grad():
                for i, (audio_data, label) in enumerate(zip(test_data, test_labels)):
                    print("test ", audio_data.shape)
                    test_predictions = self.network(audio_data)
                    test_predictions = nn.Sigmoid()(test_predictions).squeeze(1)
                    test_loss = self.loss_function(test_predictions, tensor(label).float())
                    test_predictions = nn.Sigmoid()(test_predictions)
                    test_accuracy, test_uar = accuracy_fn(test_predictions, label, self.threshold)
                    self.test_batch_loss.append(test_loss.numpy())
                    self.test_batch_accuracy.append(test_accuracy.numpy())
                    self.test_batch_uar.append(test_uar)
                print('***** Test Metrics ***** ')
                print('***** Test Metrics ***** ', file=self.log_file)
            print(
                    f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}")
            print(
                    f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}",
                    file=self.log_file)

            log_summary(self.writer, epoch, np.mean(self.batch_accuracy), np.mean(self.batch_loss),
                        np.mean(self.test_batch_accuracy), np.mean(self.test_batch_loss))

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                torch.save(self.network.state_dict(), save_path)
                print('Network successfully saved: ' + save_path)

    def test(self):
        test_data, test_labels = self.data_reader(self.data_read_path, should_batch=False, shuffle=False)
        test_predictions = self.network(test_data).detach()
        test_predictions = nn.Sigmoid()(test_predictions).squeeze(1)
        test_accuracy = accuracy_fn(test_predictions, test_labels, self.threshold)
        print(f"Accuracy: {test_accuracy}")
        print(f"Accuracy: {test_accuracy}", file=self.log_file)
