# -*- coding: utf-8 -*-
"""
@created on: 4/4/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import tensor
import time
import json
import random

from alcoaudio.networks.oneclass_net import ConvAutoEncoder
from alcoaudio.utils import file_utils
from alcoaudio.datagen.audio_feature_extractors import preprocess_data
from alcoaudio.utils.network_utils import accuracy_fn, log_summary, normalize_image
from alcoaudio.utils.data_utils import read_h5py, read_npy
from alcoaudio.datagen.augmentation_methods import librosaSpectro_to_torchTensor, time_mask, freq_mask, time_warp


class ConvAutoEncoderRunner:
    def __init__(self, args):
        self.run_name = args.run_name + '_' + str(time.time()).split('.')[0]
        self.current_run_basepath = args.network_metrics_basepath + '/' + self.run_name + '/'
        self.learning_rate = args.learning_rate
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
        self.alpha = args.alpha
        self.beta = args.beta

        self.network_metrics_basepath = args.network_metrics_basepath
        self.tensorboard_summary_path = self.current_run_basepath + args.tensorboard_summary_path
        self.network_save_path = self.current_run_basepath + args.network_save_path

        self.network_restore_path = args.network_restore_path

        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_save_interval = args.network_save_interval
        self.normalise = args.normalise_while_training
        self.dropout = args.dropout
        self.threshold = args.threshold

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        self.network = ConvAutoEncoder().to(self.device)
        self.reconstruction_loss = nn.BCEWithLogitsLoss()
        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self._min, self._max = float('inf'), -float('inf')

        if self.train_net:
            self.network.train()
            self.log_file = open(self.network_save_path + '/' + self.run_name + '.log', 'w')
            self.log_file.write(json.dumps(args))
        if self.test_net:
            print('Loading Network')
            self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
            self.network.eval()
            self.log_file = open(self.network_restore_path.replace('_40.pt', '.log'), 'a')
            print('\n\n\n********************************************************', file=self.log_file)
            print('Testing Model - ', self.network_restore_path)
            print('Testing Model - ', self.network_restore_path, file=self.log_file)
            print('********************************************************', file=self.log_file)

        self.writer = SummaryWriter(self.tensorboard_summary_path)
        print("Network config:\n", self.network)
        print("Network config:\n", self.network, file=self.log_file)

        self.batch_accuracy, self.uar = [], []

        print('Configs used:\n', json.dumps(args, indent=4))
        print('Configs used:\n', json.dumps(args, indent=4), file=self.log_file)

    def data_reader(self, data_filepath, label_filepath, train, should_batch=True, shuffle=True):
        input_data, labels = read_npy(data_filepath), read_npy(label_filepath)

        label_to_use = 1  # Sober samples
        ones_ids = [idx for idx, x in enumerate(labels) if x == label_to_use]
        input_data = input_data[ones_ids]

        if train:
            for x in input_data:
                self._min = min(np.min(x), self._min)
                self._max = max(np.max(x), self._max)

            random.shuffle(input_data)

        print('Total data ', len(input_data))
        print('Total data ', len(input_data), file=self.log_file)

        # Normalizing `input data` on train dataset's min and max values
        if self.normalise:
            input_data = (input_data - self._min) / (self._max - self._min)

        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(input_data), self.batch_size)]
            return batched_input
        else:
            return input_data

    def run_for_epoch(self, epoch, x, type):
        self.test_batch_accuracy, self.test_batch_uar, self.test_batch_ua, self.test_batch_reconstruction_loss, self.test_total_loss, audio_for_tensorboard_test = [], [], [], [], [], None
        with torch.no_grad():
            for i, audio_data in enumerate(x):
                audio_data = tensor(audio_data)
                test_reconstructions = self.network(audio_data)
                test_reconstructions = test_reconstructions.squeeze(1)
                reconstruction_test_loss = self.reconstruction_loss(test_reconstructions, audio_data)
                self.test_batch_reconstruction_loss.append(reconstruction_test_loss.numpy())
        print(f'***** {type} Metrics ***** ')
        print(f'***** {type} Metrics ***** ', file=self.log_file)
        print(f"RLoss: {np.mean(self.test_batch_reconstruction_loss)}")
        print(f"RLoss: {np.mean(self.test_batch_reconstruction_loss)}", file=self.log_file)

        log_summary(self.writer, epoch,
                    rloss=np.mean(self.test_batch_reconstruction_loss),
                    lr=self.learning_rate, type=type)

    def train(self):

        # For purposes of calculating normalized values, call this method with train data followed by test
        train_data = self.data_reader(self.data_read_path + 'train_challenge_data.npy',
                                      self.data_read_path + 'train_challenge_labels.npy',
                                      shuffle=True,
                                      train=True)
        dev_data = self.data_reader(self.data_read_path + 'dev_challenge_data.npy',
                                    self.data_read_path + 'dev_challenge_labels.npy',
                                    shuffle=False, train=False)
        test_data = self.data_reader(self.data_read_path + 'test_challenge_data.npy',
                                     self.data_read_path + 'test_challenge_labels.npy',
                                     shuffle=False, train=False)

        total_step = len(train_data)
        for epoch in range(1, self.epochs):
            self.batch_accuracy, self.batch_uar, self.batch_reconstruction_loss, self.batch_total_loss, audio_for_tensorboard_train = [], [], [], [], None
            for i, audio_data in enumerate(train_data):
                self.optimiser.zero_grad()
                audio_data = tensor(audio_data)
                if i == 0:
                    self.writer.add_graph(self.network, audio_data)
                train_reconstructions = self.network(audio_data)
                train_reconstructions = train_reconstructions.squeeze(1)
                reconstruction_loss = self.reconstruction_loss(train_reconstructions, audio_data)
                reconstruction_loss.backward()
                self.optimiser.step()
                self.batch_reconstruction_loss.append(reconstruction_loss.detach().numpy())
                if i % self.display_interval == 0:
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | RLoss: {reconstruction_loss}")
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | RLoss: {reconstruction_loss} ",
                            file=self.log_file)

            log_summary(self.writer, epoch,
                        rloss=np.mean(self.batch_reconstruction_loss), lr=self.learning_rate, type='Train')
            print('***** Overall Train Metrics ***** ')
            print('***** Overall Train Metrics ***** ', file=self.log_file)
            print(f"CLoss: RLoss: {np.mean(self.batch_reconstruction_loss)}")
            print(f"CLoss: RLoss: {np.mean(self.batch_reconstruction_loss)}", file=self.log_file)
            print('Learning rate ', self.learning_rate)
            print('Learning rate ', self.learning_rate, file=self.log_file)

            # dev data
            self.run_for_epoch(epoch, dev_data, type='Dev')

            # test data
            self.run_for_epoch(epoch, test_data, type='Test')

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                torch.save(self.network.state_dict(), save_path)
                print('Network successfully saved: ' + save_path)

    def test(self):
        test_data, test_labels = self.data_reader(self.data_read_path + 'test_data.npy',
                                                  shuffle=False,
                                                  should_batch=False)
        test_data, test_labels = test_data, test_labels
        test_predictions = self.network(test_data).detach()
        print(test_predictions)

        test_predictions = nn.Sigmoid()(test_predictions).squeeze(1)
        print(test_predictions)
        test_accuracy = accuracy_fn(test_predictions, test_labels, self.threshold)
        print(f"Accuracy: {test_accuracy}")
        print(f"Accuracy: {test_accuracy}", file=self.log_file)
