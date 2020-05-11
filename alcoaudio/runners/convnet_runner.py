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
import numpy as np
import time
import json
import random

from alcoaudio.networks.convnet import ConvNet
from alcoaudio.utils import file_utils
from alcoaudio.utils.network_utils import accuracy_fn, log_summary, custom_confusion_matrix, \
    log_conf_matrix, write_to_npy, to_tensor
from alcoaudio.utils.data_utils import read_npy
from alcoaudio.datagen.augmentation_methods import librosaSpectro_to_torchTensor, time_mask, freq_mask


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

        self.network = ConvNet().to(self.device)
        self.pos_weight = None
        self.loss_function = None
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

    def data_reader(self, data_filepath, label_filepath='', train=False, should_batch=True, shuffle=True, infer=False):

        if infer:
            input_data = read_npy(data_filepath)
            if self.normalise:
                input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
            return input_data
        else:
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

                for x in input_data:
                    self._min = min(np.min(x), self._min)
                    self._max = max(np.max(x), self._max)

                print('Data Augmentation starts . . .')
                print('Data Augmentation starts . . .', file=self.log_file)
                label_to_augment = 1
                amount_to_augment = 1
                ones_ids = [idx for idx, x in enumerate(labels) if x == label_to_augment]
                random_idxs = random.choices(ones_ids,
                                             k=int(len(ones_ids) * amount_to_augment))
                data_to_augment = input_data[random_idxs]
                augmented_data = []
                augmented_labels = []
                for x in data_to_augment:
                    x = librosaSpectro_to_torchTensor(x)
                    x = random.choice([time_mask, freq_mask])(x)[0].numpy()
                    augmented_data.append(x), augmented_labels.append(label_to_augment)

                input_data = np.concatenate((input_data, augmented_data))
                labels = np.concatenate((labels, augmented_labels))

                print('Data Augmentation done . . .')
                print('Data Augmentation done . . .', file=self.log_file)

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
        predictions_dict = {"tp": [], "fp": [], "tn": [], "fn": []}
        predictions = []
        self.test_batch_loss, self.test_batch_accuracy, self.test_batch_uar, self.test_batch_ua, audio_for_tensorboard_test = [], [], [], [], None
        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                label = to_tensor(label, device=self.device).float()
                audio_data = to_tensor(audio_data, device=self.device)
                test_predictions = self.network(audio_data).squeeze(1)
                test_loss = self.loss_function(test_predictions, label)
                test_predictions = nn.Sigmoid()(test_predictions)
                predictions.append(test_predictions.numpy())
                test_accuracy, test_uar = accuracy_fn(test_predictions, label, self.threshold)
                self.test_batch_loss.append(test_loss.numpy())
                self.test_batch_accuracy.append(test_accuracy)
                self.test_batch_uar.append(test_uar)
                tp, fp, tn, fn = custom_confusion_matrix(test_predictions, label, threshold=self.threshold)
                predictions_dict['tp'].extend(tp)
                predictions_dict['fp'].extend(fp)
                predictions_dict['tn'].extend(tn)
                predictions_dict['fn'].extend(fn)

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
        log_conf_matrix(self.writer, epoch, predictions_dict=predictions_dict, type=type)

        y = [element for sublist in y for element in sublist]
        predictions = [element for sublist in predictions for element in sublist]
        write_to_npy(filename=self.debug_filename, predictions=predictions, labels=y, epoch=epoch, accuracy=np.mean(
                self.test_batch_accuracy), loss=np.mean(self.test_batch_loss), uar=np.mean(self.test_batch_uar),
                     lr=self.optimiser.state_dict()['param_groups'][0]['lr'], predictions_dict=predictions_dict,
                     type=type)

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

        # For the purposes of assigning pos weight on the fly we are initializing the cost function here
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=to_tensor(self.pos_weight, device=self.device))

        total_step = len(train_data)
        for epoch in range(1, self.epochs):
            self.network.train()
            self.batch_loss, self.batch_accuracy, self.batch_uar, audio_for_tensorboard_train = [], [], [], None
            for i, (audio_data, label) in enumerate(zip(train_data, train_labels)):
                self.optimiser.zero_grad()
                label = to_tensor(label, device=self.device).float()
                audio_data = to_tensor(audio_data, device=self.device).float()
                if i == 0:
                    self.writer.add_graph(self.network, audio_data)
                predictions = self.network(audio_data).squeeze(1)
                loss = self.loss_function(predictions, label)
                loss.backward()
                self.optimiser.step()
                predictions = nn.Sigmoid()(predictions)
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
        test_data, test_labels = self.data_reader(self.data_read_path + 'test_challenge_data.npy',
                                                  self.data_read_path + 'test_challenge_labels.npy',
                                                  shuffle=False, train=False)
        test_predictions = self.network(test_data).squeeze(1)
        test_predictions = nn.Sigmoid()(test_predictions)
        test_accuracy, test_uar = accuracy_fn(test_predictions, test_labels, self.threshold)
        print(f"Accuracy: {test_accuracy} | UAR: {test_uar}")
        print(f"Accuracy: {test_accuracy} | UAR: {test_uar}", file=self.log_file)

    def infer(self, data_file):
        test_data = self.data_reader(data_file, shuffle=False, train=False, infer=True)
        test_predictions = self.network(test_data).squeeze(1)
        test_predictions = nn.Sigmoid()(test_predictions)

        return test_predictions
