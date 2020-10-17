# -*- coding: utf-8 -*-
"""
@created on: 2/27/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import pandas as pd
import argparse
import json
import numpy as np

from alcoaudio.utils.class_utils import AttributeDict
from alcoaudio.datagen.audio_feature_extractors import preprocess_data, preprocess_data_images, \
    remove_silent_parts_from_audio
from alcoaudio.utils.data_utils import save_h5py, save_npy, save_csv


def parse():
    parser = argparse.ArgumentParser(description="alcoaudio_data_processor")
    parser.add_argument('--configs_file', type=str)
    args = parser.parse_args()
    return args


class DataProcessor:
    def __init__(self, args):
        self.base_path = args.audio_basepath
        self.train_data_file = args.train_data_file
        self.dev_data_file = args.dev_data_file
        self.test_data_file = args.test_data_file
        self.normalise = args.normalise_while_creating
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_save_path = args.data_save_path
        self.image_save_path = args.image_data_save_path
        self.method = args.data_processing_method

    def process_audio_and_save_h5py(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file)
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data(self.base_path, df['WAV_PATH'].values, df['label'].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        save_h5py(data, self.data_save_path + '/' + filename_to_save + '_data.h5')
        save_h5py(labels, self.data_save_path + '/' + filename_to_save + '_labels.h5')

    def process_audio_and_save_npy(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file)
        print('Read a ')
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data(self.base_path, df['WAV_PATH'].values, df['label'].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        save_npy(data, self.data_save_path + '/' + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + '/' + filename_to_save + '_labels.npy')

    def process_audio_and_save_npy_challenge(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file, header=None, delimiter='\t')
        print('Number of audio files ', len(df))
        if shuffle:
            df = df.sample(frac=1)

        # Converting 'A' to 1 and 'N' to 0 - according to Challenge's binary decision.
        # Refer Challenge's Readme for further information
        df[1] = df[1].apply(lambda x: 1 if x == 'A' else 0)

        # Irregular use of extensions in data, so handling it here
        df[0] = df[0].apply(lambda x: x.replace('WAV', 'wav'))
        data, labels = preprocess_data(self.base_path, df[0].values, df[1].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap, self.method)
        print('Number of audio files after processing ', len(data))
        save_npy(data, self.data_save_path + '/' + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + '/' + filename_to_save + '_labels.npy')
        del data
        del labels

    def process_audio_and_save_csv(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file, header=None, delimiter='\t')
        if shuffle:
            df = df.sample(frac=1)

        # Converting 'A' to 1 and 'N' to 0 - according to Challenge's binary decision.
        # Refer Challenge's Readme for further information
        df[1] = df[1].apply(lambda x: 1 if x == 'A' else 0)

        # Irregular use of extensions in data, so handling it here
        df[0] = df[0].apply(lambda x: x.replace('WAV', 'wav'))

        data, labels = preprocess_data_images(self.base_path, self.image_save_path, df[0].values,
                                              df[1].values,
                                              self.normalise,
                                              self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        concat_data = np.concatenate((np.array([data]).T, np.array([labels]).T), axis=1)
        save_csv(concat_data, columns=["spectrogram_path", "labels"], filename=
        self.data_save_path + '/' + filename_to_save + '_data_melfilter_specs.csv')

    def silent_parts_removal(self, data_file):
        df = pd.read_csv(data_file, header=None, delimiter='\t')
        print('Number of audio files ', len(df))

        # Irregular use of extensions in data, so handling it here
        df[0] = df[0].apply(lambda x: x.replace('WAV', 'wav'))
        remove_silent_parts_from_audio(self.base_path, df[0].values, self.sampling_rate)

    def run(self):
        print('Started processing train data . . .')
        self.process_audio_and_save_npy_challenge(self.train_data_file,
                                                  filename_to_save='train_challenge_with_d1_mel_power_to_db_fnot_single')
        print('Started processing dev data . . .')
        self.process_audio_and_save_npy_challenge(self.dev_data_file,
                                                  filename_to_save='dev_challenge_with_d1_mel_power_to_db_fnot_single')
        print('Started processing test data . . .')
        self.process_audio_and_save_npy_challenge(self.test_data_file,
                                                  filename_to_save='test_challenge_with_d1_mel_power_to_db_fnot_single')

        # print('Started processing train data . . .')
        # self.silent_parts_removal(self.train_data_file)
        # print('Started processing dev data . . .')
        # self.silent_parts_removal(self.dev_data_file)
        # print('Started processing test data . . .')
        # self.silent_parts_removal(self.test_data_file)

    def run_images(self):
        print('Started processing train data . . .')
        self.process_audio_and_save_csv(self.train_data_file,
                                        filename_to_save='train_challenge_with_d1_mel_images')
        print('Started processing dev data . . .')
        self.process_audio_and_save_csv(self.dev_data_file,
                                        filename_to_save='dev_challenge_with_d1_mel_images')
        print('Started processing test data . . .')
        self.process_audio_and_save_csv(self.test_data_file,
                                        filename_to_save='test_challenge_with_d1_mel_images')


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)

    processor = DataProcessor(configs)
    print(configs)
    processor.run()
