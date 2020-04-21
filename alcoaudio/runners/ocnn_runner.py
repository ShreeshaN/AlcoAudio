# -*- coding: utf-8 -*-
"""
@created on: 4/19/20,
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
import cv2
import random
import torchvision
import random

from alcoaudio.networks.oneclass_net import OneClassNN, ConvAutoEncoder
from alcoaudio.utils import file_utils
from alcoaudio.datagen.audio_feature_extractors import preprocess_data
from alcoaudio.utils.network_utils import accuracy_fn_ocnn, calc_average_class_score, log_summary_ocnn, normalize_image, \
    custom_confusion_matrix, \
    log_conf_matrix, write_to_npy
from alcoaudio.utils.data_utils import read_h5py, read_npy
from alcoaudio.datagen.augmentation_methods import librosaSpectro_to_torchTensor, time_mask, freq_mask, time_warp


class OCNNRunner:
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
        self.c = tensor(0.0)
        self.r = tensor(0.0)
        self.nu = None  # Updated in data_reader()
        self.eps = 0.1

        self.network_metrics_basepath = args.network_metrics_basepath
        self.tensorboard_summary_path = self.current_run_basepath + args.tensorboard_summary_path
        self.network_save_path = self.current_run_basepath + args.network_save_path

        self.network_restore_path = args.network_restore_path

        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_save_interval = args.network_save_interval
        self.normalise = args.normalise_while_training
        self.dropout = args.dropout
        self.threshold = args.threshold
        self.debug_filename = self.current_run_basepath + '/' + args.debug_filename

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        self.cae_network = ConvAutoEncoder()
        self.cae_model_restore_path = args.cae_model_restore_path
        self.cae_network.load_state_dict(torch.load(self.cae_model_restore_path, map_location=self.device))
        self.cae_network.eval()

        self.network = OneClassNN().to(self.device)
        self.learning_rate_decay = args.learning_rate_decay
        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=self.learning_rate_decay)

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

        self.batch_loss, self.batch_accuracy, self.uar = [], [], []

        print('Configs used:\n', json.dumps(args, indent=4))
        print('Configs used:\n', json.dumps(args, indent=4), file=self.log_file)

    def data_reader(self, data_filepath, label_filepath, train, should_batch=True, shuffle=True):
        input_data, labels = read_npy(data_filepath), read_npy(label_filepath)

        if train:
            ones_len = len([x for x in labels if x == 1])
            zeros_len = len(labels) - ones_len

            # nu declared in init, initialized here based on the number of anomalies.
            # Here intoxicated samples are considered anomalies
            self.nu = ones_len / zeros_len

            for x in input_data:
                self._min = min(np.min(x), self._min)
                self._max = max(np.max(x), self._max)

            data = [(x, y) for x, y in zip(input_data, labels)]
            random.shuffle(data)
            input_data, labels = np.array([x[0] for x in data]), [x[1] for x in data]

        print('Total data ', len(input_data))
        print('Event rate', sum(labels) / len(labels))
        print(np.array(input_data).shape, np.array(labels).shape)

        print('Total data ', len(input_data), file=self.log_file)
        print('Event rate', sum(labels) / len(labels), file=self.log_file)
        print(np.array(input_data).shape, np.array(labels).shape, file=self.log_file)

        print('Min max values used for normalisation ', self._min, self._max)
        print('Min max values used for normalisation ', self._min, self._max, file=self.log_file)

        # Normalizing `input data` on train dataset's min and max values
        if self.normalise:
            input_data = (input_data - self._min) / (self._max - self._min)

        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(input_data), self.batch_size)]
            batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
            return batched_input, batched_labels
        else:
            return input_data, labels

    def run_for_epoch(self, epoch, x, y, type):
        self.test_batch_loss, self.test_batch_accuracy, self.test_batch_uar, self.test_scores_list, audio_for_tensorboard_test = [], [], [], [], None
        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                label = tensor(label).float()
                audio_data = tensor(audio_data)
                latent_vector = self.get_latent_vector(audio_data)
                test_predictions, w, v = self.network(latent_vector)
                test_loss = self.loss_function(test_predictions, w, v)
                test_scores = self.calc_scores(test_predictions)
                test_accuracy, test_uar = accuracy_fn_ocnn(test_scores, label)
                self.test_scores_list.extend(test_scores)
                self.test_batch_loss.append(test_loss.numpy())
                self.test_batch_accuracy.append(test_accuracy.numpy())
                self.test_batch_uar.append(test_uar)

        print(f'***** {type} Metrics ***** ')
        print(f'***** {type} Metrics ***** ', file=self.log_file)
        print(
                f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}")
        print(
                f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}",
                file=self.log_file)
        y = [item for sublist in y for item in sublist]
        pos_score, neg_score = calc_average_class_score(tensor(self.test_scores_list), y)
        log_summary_ocnn(self.writer, epoch, accuracy=np.mean(self.test_batch_accuracy),
                         loss=np.mean(self.test_batch_loss),
                         uar=np.mean(self.test_batch_uar), lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
                         r=self.r, positive_class_score=pos_score, negative_class_score=neg_score,
                         type=type)

    def get_latent_vector(self, audio_data):
        latent_filter_maps, _, _ = self.cae_network.encoder(audio_data)
        latent_vector = latent_filter_maps.view(-1, latent_filter_maps.size()[1:].numel())
        return latent_vector.detach()

    def loss_function(self, y_pred, w, v):
        # term1 = 0.5 * torch.sum(w ** 2)
        # term2 = 0.5 * torch.sum(v ** 2)
        # term3 = 1 / self.nu * torch.mean(torch.max(0, self.r - y_pred))
        # term4 = -1 * self.r

        term3 = self.r ** 2 + torch.sum(torch.max(tensor(0.0), (y_pred - self.c) ** 2 - self.r ** 2), axis=1)
        term3 = 1 / self.nu * torch.mean(term3)
        return term3

    def calc_scores(self, outputs):
        scores = torch.sum((outputs - self.c) ** 2, axis=1)
        return scores

    def update_r_and_c(self, outputs):
        centroids = torch.mean(outputs, axis=0)

        centroids[(abs(centroids) < self.eps) & (centroids < 0)] = -self.eps
        centroids[(abs(centroids) < self.eps) & (centroids > 0)] = self.eps

        scores = torch.sum((outputs - centroids) ** 2, axis=1)
        sorted_scores, _ = torch.sort(scores)
        self.r = np.percentile(sorted_scores, self.nu * 100)  # Updating the value of self.r
        self.c = centroids

    def initalize_c_and_r(self, train_x):
        predictions_list = []
        for batch in train_x:
            batch = tensor(batch)
            latent_vec = self.get_latent_vector(batch)
            preds, _, _ = self.network(latent_vec)
            predictions_list.extend(preds.detach().numpy())
        self.update_r_and_c(tensor(predictions_list))

    def train(self):

        # For purposes of calculating normalized values, call this method with train data followed by test
        train_data, train_labels = self.data_reader(self.data_read_path + 'train_challenge_with_d1_data.npy',
                                                    self.data_read_path + 'train_challenge_with_d1_labels.npy',
                                                    shuffle=True,
                                                    train=True)
        dev_data, dev_labels = self.data_reader(self.data_read_path + 'dev_challenge_with_d1_data.npy',
                                                self.data_read_path + 'dev_challenge_with_d1_labels.npy',
                                                shuffle=False, train=False)
        test_data, test_labels = self.data_reader(self.data_read_path + 'test_challenge_data.npy',
                                                  self.data_read_path + 'test_challenge_labels.npy',
                                                  shuffle=False, train=False)

        total_step = len(train_data)
        train_labels_flattened = [item for sublist in train_labels for item in sublist]

        # Initialize c and r which is declared in init, on entire train data
        self.initalize_c_and_r(train_data)

        for epoch in range(1, self.epochs):
            self.batch_loss, self.batch_accuracy, self.batch_uar, self.total_predictions, self.total_scores, audio_for_tensorboard_train = [], [], [], [], [], None
            for i, (audio_data, label) in enumerate(zip(train_data, train_labels)):
                self.optimiser.zero_grad()
                label = tensor(label).float()
                audio_data = tensor(audio_data)
                latent_vector = self.get_latent_vector(audio_data)
                # if i == 0 and epoch == 1:
                #     self.writer.add_graph(self.network, tensor(sample_data))
                predictions, w, v = self.network(latent_vector)
                loss = self.loss_function(predictions, w, v)
                loss.backward()
                self.optimiser.step()

                self.total_predictions.extend(predictions.detach().numpy())

                scores = self.calc_scores(predictions)
                self.total_scores.extend(scores)
                accuracy, uar = accuracy_fn_ocnn(scores, label)
                self.batch_loss.append(loss.detach().numpy())
                self.batch_accuracy.append(accuracy)
                self.batch_uar.append(uar)

                if i % self.display_interval == 0:
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}")
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}",
                            file=self.log_file)

            pos_class_score, neg_class_score = calc_average_class_score(tensor(self.total_scores),
                                                                        train_labels_flattened)
            self.update_r_and_c(tensor(self.total_predictions))  # Update value of r and c after every epoch

            # Decay learning rate
            self.scheduler.step(epoch=epoch)
            log_summary_ocnn(self.writer, epoch, accuracy=np.mean(self.batch_accuracy),
                             loss=np.mean(self.batch_loss),
                             uar=np.mean(self.batch_uar), lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
                             r=self.r, positive_class_score=pos_class_score, negative_class_score=neg_class_score,
                             type='Train')
            print('***** Overall Train Metrics ***** ')
            print('***** Overall Train Metrics ***** ', file=self.log_file)
            print(
                    f"Loss: {np.mean(self.batch_loss)} | Accuracy: {np.mean(self.batch_accuracy)} | UAR: {np.mean(self.batch_uar)} ")
            print(
                    f"Loss: {np.mean(self.batch_loss)} | Accuracy: {np.mean(self.batch_accuracy)} | UAR: {np.mean(self.batch_uar)} ",
                    file=self.log_file)
            print('Learning rate ', self.optimiser.state_dict()['param_groups'][0]['lr'])
            print('Learning rate ', self.optimiser.state_dict()['param_groups'][0]['lr'], file=self.log_file)

            # dev data
            self.run_for_epoch(epoch, dev_data, dev_labels, type='Dev')

            # test data
            self.run_for_epoch(epoch, test_data, test_labels, type='Test')

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
        test_accuracy = accuracy_fn_ocnn(test_predictions, test_labels, self.threshold)
        print(f"Accuracy: {test_accuracy}")
        print(f"Accuracy: {test_accuracy}", file=self.log_file)
