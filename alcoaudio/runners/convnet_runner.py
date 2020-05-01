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
import cv2
import random
import torchvision
import random

from alcoaudio.networks.convnet import ConvNet
from alcoaudio.utils import file_utils
from alcoaudio.datagen.audio_feature_extractors import preprocess_data
from alcoaudio.utils.network_utils import accuracy_fn_two_class_nn, log_summary, normalize_image, \
    custom_confusion_matrix, \
    log_conf_matrix, write_to_npy
from alcoaudio.utils.data_utils import read_h5py, read_npy
from alcoaudio.datagen.augmentation_methods import librosaSpectro_to_torchTensor, time_mask, freq_mask, time_warp


class ConvNetRunner:
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

        self.weights = np.load(args.keras_model_weights, allow_pickle=True)
        self.network = None
        self.network = ConvNet().to(self.device)
        self.pos_weight = None
        self.loss_function = nn.NLLLoss()
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
        # data = pd.read_csv(data_file)[:50]
        # if shuffle:
        #     data = data.sample(frac=1)
        # input_data, labels = preprocess_data(self.audio_basepath, data['WAV_PATH'].values, data['label'].values,
        #                                      normalise=normalise, sample_size_in_seconds=self.sample_size_in_seconds,
        #                                      sampling_rate=self.sampling_rate, overlap=self.overlap)
        input_data, labels = read_npy(data_filepath), read_npy(label_filepath)

        if train:

            print('Original data size - before Augmentation')
            print('Original data size - before Augmentation', file=self.log_file)
            print('Total data ', len(input_data))
            print('Event rate', sum(labels) / len(labels))
            print(np.array(input_data).shape, np.array(labels).shape)

            print('Total data ', len(input_data), file=self.log_file)
            print('Event rate', sum(labels) / len(labels), file=self.log_file)
            print(np.array(input_data).shape, np.array(labels).shape, file=self.log_file)

            # Under-sampling train data. Balancing the classes
            ones_idx, zeros_idx = [idx for idx, label in enumerate(labels) if label == 1], [idx for idx, label in
                                                                                            enumerate(labels) if
                                                                                            label == 0]
            zeros_idx = zeros_idx[:len(ones_idx)]
            ids = ones_idx + zeros_idx
            input_data, labels = input_data[ids], labels[ids]

            for x in input_data:
                self._min = min(np.min(x), self._min)
                self._max = max(np.max(x), self._max)

            # print('Data Augmentation starts . . .')
            # print('Data Augmentation starts . . .', file=self.log_file)
            # label_to_augment = 1
            # amount_to_augment = 1
            # ones_ids = [idx for idx, x in enumerate(labels) if x == label_to_augment]
            # random_idxs = random.choices(ones_ids,
            #                              k=int(len(ones_ids) * amount_to_augment))
            # data_to_augment = input_data[random_idxs]
            # augmented_data = []
            # augmented_labels = []
            # for x in data_to_augment:
            #     x = librosaSpectro_to_torchTensor(x)
            #     x = random.choice([time_mask, freq_mask])(x)[0].numpy()
            #     # x = time_warp(x)[0].numpy()
            #     augmented_data.append(x), augmented_labels.append(label_to_augment)
            #
            # input_data = np.concatenate((input_data, augmented_data))
            # labels = np.concatenate((labels, augmented_labels))
            #
            # print('Data Augmentation done . . .')
            # print('Data Augmentation done . . .', file=self.log_file)

            data = [(x, y) for x, y in zip(input_data, labels)]
            random.shuffle(data)
            input_data, labels = np.array([x[0] for x in data]), [x[1] for x in data]

            # Initialize pos_weight based on training data
            self.pos_weight = len([x for x in labels if x == 0]) / len([x for x in labels if x == 1])
            print('Pos weight for the train data - ', self.pos_weight)
            print('Pos weight for the train data - ', self.pos_weight, file=self.log_file)

        print('Total data ', len(input_data))
        print('Event rate', sum(labels) / len(labels))
        print(np.array(input_data).shape, np.array(labels).shape)

        print('Total data ', len(input_data), file=self.log_file)
        print('Event rate', sum(labels) / len(labels), file=self.log_file)
        print(np.array(input_data).shape, np.array(labels).shape, file=self.log_file)

        print('Min max values used for normalisation ', self._min, self._max)
        print('Min max values used for normalisation ', self._min, self._max, file=self.log_file)

        # Convert binary label into 2 class
        labels = np.eye(self.num_classes)[np.array(labels).astype(int)]

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
        self.network.eval()
        predictions = []
        self.test_batch_loss, self.test_batch_accuracy, self.test_batch_uar, self.test_batch_ua, audio_for_tensorboard_test = [], [], [], [], None
        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                label = tensor(label).float()
                test_predictions = self.network(audio_data)
                test_loss = self.loss_function(test_predictions, label)
                predictions.append(test_predictions.numpy())
                test_accuracy, test_uar = accuracy_fn_two_class_nn(test_predictions, label)
                self.test_batch_loss.append(test_loss.numpy())
                self.test_batch_accuracy.append(test_accuracy)
                self.test_batch_uar.append(test_uar)

        print(f'***** {type} Metrics ***** ')
        print(f'***** {type} Metrics ***** ', file=self.log_file)
        print(
                f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}")
        print(
                f"Loss: {np.mean(self.test_batch_loss)} | Accuracy: {np.mean(self.test_batch_accuracy)} | UAR: {np.mean(self.test_batch_uar)}",
                file=self.log_file)

        log_summary(self.writer, epoch, accuracy=np.mean(self.test_batch_accuracy),
                    loss=np.mean(self.test_batch_loss),
                    uar=np.mean(self.test_batch_uar), lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
                    type=type)

        y = [element for sublist in y for element in sublist]
        predictions = [element for sublist in predictions for element in sublist]
        write_to_npy(filename=self.debug_filename, predictions=predictions, labels=y, epoch=epoch, accuracy=np.mean(
                self.test_batch_accuracy), loss=np.mean(self.test_batch_loss), uar=np.mean(self.test_batch_uar),
                     lr=self.optimiser.state_dict()['param_groups'][0]['lr'], type=type)

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
        for epoch in range(1, self.epochs):
            self.network.train()
            self.batch_loss, self.batch_accuracy, self.batch_uar, audio_for_tensorboard_train = [], [], [], None
            for i, (audio_data, label) in enumerate(zip(train_data, train_labels)):
                self.optimiser.zero_grad()
                label = tensor(label).float()
                if i == 0:
                    self.writer.add_graph(self.network, tensor(audio_data))
                predictions = self.network(audio_data)
                loss = self.loss_function(predictions, label)
                loss.backward()
                self.optimiser.step()
                accuracy, uar = accuracy_fn_two_class_nn(predictions, label)
                self.batch_loss.append(loss.detach().numpy())
                self.batch_accuracy.append(accuracy)
                self.batch_uar.append(uar)

                if i % self.display_interval == 0:
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}")
                    print(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {loss} | Accuracy: {accuracy} | UAR: {uar}",
                            file=self.log_file)
            # Decay learning rate
            self.scheduler.step(epoch=epoch)
            log_summary(self.writer, epoch, accuracy=np.mean(self.batch_accuracy),
                        loss=np.mean(self.batch_loss),
                        uar=np.mean(self.batch_uar), lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
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
        test_accuracy = accuracy_fn_two_class_nn(test_predictions, test_labels)
        print(f"Accuracy: {test_accuracy}")
        print(f"Accuracy: {test_accuracy}", file=self.log_file)

# def test():
#     def read_data():
#         batch_size = 32
#         test_data, test_labels = \
#             np.load('/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_data.npy'), \
#             np.load('/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy')
#         min_, max_ = -80, 9.536743e-07
#         for i, x in enumerate(test_data):
#             # min_, max_ = np.min(x), np.max(x)
#             test_data[i] = (x - min_) / (max_ - min_)
#         bi = [test_data[pos:pos + batch_size] for pos in
#               range(0, len(test_data), batch_size)]
#         bl = [test_labels[pos:pos + batch_size] for pos in range(0, len(test_labels), batch_size)]
#         return bi, bl
#
#     def read_model():
#         network_restore_path = '/Users/badgod/badgod_documents/Alco_audio/server_data/2d/emotion_alco_audio_challenge_data_cnn_reproducing_results_1588198928/alco_trained_models/emotion_alco_audio_challenge_data_cnn_reproducing_results_1588198928_9.pt'
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         from alcoaudio.networks.convnet import ConvNet
#         network = ConvNet().to(device)
#         network.load_state_dict(torch.load(network_restore_path, map_location=device))
#         # network.eval()
#         return network
#
#     bi, bl = read_data()
#     print('Data read done')
#     network = read_model()
#
#     flattened_preds, flattened_labels = [], []
#
#     print('starting prediction')
#     with torch.no_grad():
#         for i, (x, y) in enumerate(zip(bi, bl)):
#             test_predictions = network(x).detach()
#             test_predictions = nn.Sigmoid()(test_predictions).squeeze(1)
#             flattened_preds.extend(test_predictions)
#             flattened_labels.extend(y)
#             print(f'Running on {i}/{len(bl)} | Mean pred: {np.mean(test_predictions.numpy())}')
#
#     print(np.array(flattened_preds).mean())
#     print('saving file')
#     np.save('/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_data_preds.npy', flattened_preds)
#
#     for threshold in range(1, 10):
#         test_accuracy, test_uar = accuracy_fn(tensor(flattened_preds), tensor(flattened_labels), threshold / 10)
#         print(f"Threshold: {threshold / 10} || Accuracy: {test_accuracy} | UAR: {test_uar}")

# test()
