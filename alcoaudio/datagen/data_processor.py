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
import time
from alcoaudio.utils.class_utils import AttributeDict
from alcoaudio.datagen.audio_feature_extractors import preprocess_data
from alcoaudio.utils.data_utils import save_h5py, save_npy


def parse():
    parser = argparse.ArgumentParser(description="alcoaudio_data_processor")
    parser.add_argument('--configs_file', type=str)
    args = parser.parse_args()
    return args


class DataProcessor:
    def __init__(self, args):
        self.base_path = args.audio_basepath
        self.train_data_file = args.train_data_file
        self.test_data_file = args.test_data_file
        self.normalise = args.normalise
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_save_path = args.data_save_path

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
        df = pd.read_csv(data_file)[:30]
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data(self.base_path, df['WAV_PATH'].values, df['label'].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        save_npy(data, self.data_save_path + '/' + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + '/' + filename_to_save + '_labels.npy')

    def run(self):
        self.process_audio_and_save_npy(self.train_data_file, filename_to_save='train')
        self.process_audio_and_save_npy(self.test_data_file, filename_to_save='test')


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)

    processor = DataProcessor(configs)
    processor.run()
