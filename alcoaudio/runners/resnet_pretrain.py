# # -*- coding: utf-8 -*-
# """
# @created on: 11/7/20,
# @author: Shreesha N,
# @version: v0.0.1
# @system name: badgod
# Description:
#
# ..todo::
#

import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from alcoaudio.datagen.augmentation_methods import librosaSpectro_to_torchTensor, time_mask, freq_mask
from alcoaudio.utils import file_utils
from alcoaudio.utils.data_utils import read_npy
from alcoaudio.utils.logger import Logger
from alcoaudio.utils.network_utils import to_tensor, to_numpy

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)


class ResNetRunner:
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
        self.data_augment = args.data_augment
        self.dropout = args.dropout
        self.threshold = args.threshold
        self.debug_filename = self.current_run_basepath + '/' + args.debug_filename

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        self.pos_weight = None
        self.learning_rate_decay = args.learning_rate_decay

        self._min, self._max = float('inf'), -float('inf')
        self.logger = Logger(name=self.run_name, log_path=self.network_save_path).get_logger()
        self.logger.info(str(json.dumps(args, indent=4)))

        self.writer = SummaryWriter(self.tensorboard_summary_path)
        self.batch_loss, self.batch_accuracy, self.uar = [], [], []
        self.logger.info(f'Configs used:\n{json.dumps(args, indent=4)}')
        self.data_loaders = {}
        self.dataset_sizes = {}

    def data_reader(self, data_filepath, label_filepath, jitter_filepath, train, type, should_batch=True, shuffle=True,
                    infer=False):
        if infer:
            pass
        else:
            input_data, labels, jitter = read_npy(data_filepath), read_npy(label_filepath), read_npy(
                    jitter_filepath)

            if train:
                self.logger.info(f'Original data size - before Augmentation')
                self.logger.info(f'Total data {str(len(input_data))}')
                self.logger.info(f'Event rate {str(sum(labels) / len(labels))}')
                self.logger.info(
                        f'Input data shape:{np.array(input_data).shape} | Output data shape:{np.array(labels).shape}')

                for x in input_data:
                    self._min = min(np.min(x), self._min)
                    self._max = max(np.max(x), self._max)
                self._mean, self._std = np.mean(input_data), np.std(input_data)
                self._jmean, self._jstd = np.mean(jitter), np.std(jitter)
                self._jmin, self._jmax = np.min(jitter), np.max(jitter)

                if self.data_augment:
                    self.logger.info(f'Data Augmentation starts . . .')
                    label_to_augment = 1
                    amount_to_augment = 1.3
                    ones_ids = [idx for idx, x in enumerate(labels) if x == label_to_augment]
                    random_idxs = random.choices(ones_ids,
                                                 k=int(len(ones_ids) * amount_to_augment))
                    data_to_augment = input_data[random_idxs]
                    augmented_data, jitter_augmented_data = [], []
                    augmented_labels = []
                    for x in data_to_augment:
                        x = librosaSpectro_to_torchTensor(x)
                        x = random.choice([time_mask, freq_mask])(x)[0].numpy()
                        augmented_data.append(x), augmented_labels.append(label_to_augment)

                    # Jitter and shimmer
                    # jitter_augmented_data, jitter_labels = BorderlineSMOTE().fit_resample(X=jitter, y=labels)
                    #
                    # assert np.mean(jitter_labels[len(jitter):][
                    #                :len(augmented_data)]) == 1, 'Issue with Jitter Shimmer Augmentation'
                    #
                    # jitter = np.concatenate((jitter, jitter_augmented_data[len(jitter):][:len(augmented_data)]))
                    input_data = np.concatenate((input_data, augmented_data))
                    labels = np.concatenate((labels, augmented_labels))

                    # Temp fix
                    # input_data = input_data[:len(jitter)]
                    # labels = labels[:len(jitter)]

                    # assert len(jitter) == len(
                    #         input_data), "Input data and Jitter Shimmer augmentations don't match in length"

                    self.logger.info(f'Data Augmentation done . . .')

                # data = [(x, y, z) for x, y, z in zip(input_data, labels, jitter)]
                # random.shuffle(data)
                # input_data, labels, jitter = np.array([x[0] for x in data]), [x[1] for x in data], np.array(
                #         [x[2] for x in data])

                data = [(x, y) for x, y in zip(input_data, labels)]
                random.shuffle(data)
                input_data, labels = np.array([x[0] for x in data]), [x[1] for x in data]

                # Initialize pos_weight based on training data
                self.pos_weight = len([x for x in labels if x == 0]) / len([x for x in labels if x == 1])
                self.logger.info(f'Pos weight for the train data - {self.pos_weight}')

            self.logger.info(f'Total data {str(len(input_data))}')
            self.logger.info(f'Event rate {str(sum(labels) / len(labels))}')
            self.logger.info(
                    f'Input data shape:{np.array(input_data).shape} | Output data shape:{np.array(labels).shape}')

            self.logger.info(f'Min max values used for normalisation {self._min, self._max}')
            self.logger.info(f'Min max values used for normalisation {self._min, self._max}')

            # Normalizing `input data` on train dataset's min and max values
            if self.normalise:
                input_data = (input_data - self._min) / (self._max - self._min)
                input_data = (input_data - self._mean) / self._std

                # jitter = (jitter - self._jmin) / (self._jmax - self._jmin)
                # jitter = (jitter - self._jmean) / self._jstd

            self.dataset_sizes[type] = len(input_data)
            return DataLoader(
                    TensorDataset(torch.Tensor(input_data).unsqueeze(1).repeat(1, 3, 1, 1),
                                  torch.Tensor(labels)),
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(list([x for x in range(10)])))

    #

    # plt.ion()  # interactive mode
    #
    # # Data augmentation and normalization for training
    # # Just normalization for validation
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    #
    # # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, num_workers=1
    # #                                               , sampler=torch.utils.data.SubsetRandomSampler(
    # #             list([x for x in range(100)]))
    # #                                               )
    # #                for x in ['train', 'val']}
    # # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # # class_names = image_datasets['train'].classes
    # #
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):

        def calc_uar(y_true, y_pred):
            uar = recall_score(to_numpy(y_true), to_numpy(y_pred), average='macro')
            return uar

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'dev', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                total_preds, total_labels = [], []
                # Iterate over data.
                for inputs, labels in self.data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    total_labels.extend(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).squeeze(1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    binary_preds = torch.where(outputs > to_tensor(0.5), to_tensor(1), to_tensor(0))
                    total_preds.extend(binary_preds)

                    running_corrects += torch.sum(binary_preds == labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                epoch_uar = calc_uar(total_preds, total_labels)

                print('{} Loss: {:.4f} Acc: {:.4f} UAR: {}'.format(
                        phase, epoch_loss, epoch_acc, epoch_uar))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def train(self):
        train_inp_file, train_out_file, train_jitter_file = 'train_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_data.npy', 'train_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_labels.npy', 'train_challenge_with_shimmer_jitter.npy'
        dev_inp_file, dev_out_file, dev_jitter_file = 'dev_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_data.npy', 'dev_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_labels.npy', 'dev_challenge_with_shimmer_jitter.npy'
        test_inp_file, test_out_file, test_jitter_file = 'test_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_data.npy', 'test_challenge_with_d1_mel_power_to_db_fnot_zr_crossing_labels.npy', 'test_challenge_with_shimmer_jitter.npy'

        self.data_loaders['train'] = self.data_reader(
                self.data_read_path + train_inp_file,
                self.data_read_path + train_out_file,
                self.data_read_path + train_jitter_file,
                shuffle=True,
                train=True, type='train')
        # a,b = next(iter(data_loaders['train']))
        self.logger.info(f'Reading dev file {dev_inp_file, dev_out_file}')
        self.data_loaders['dev'] = self.data_reader(self.data_read_path + dev_inp_file,
                                                    self.data_read_path + dev_out_file,
                                                    self.data_read_path + dev_jitter_file,
                                                    shuffle=False, train=False, type='dev')
        self.logger.info(f'Reading test file {test_inp_file, test_out_file}')
        self.data_loaders['test'] = self.data_reader(self.data_read_path + test_inp_file,
                                                     self.data_read_path + test_out_file,
                                                     self.data_read_path + test_jitter_file,
                                                     shuffle=False, train=False, type='test')

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 1)
        model_ft = model_ft.to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=to_tensor(self.pos_weight, device=self.device))

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                         num_epochs=25)
