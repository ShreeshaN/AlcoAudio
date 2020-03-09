# -*- coding: utf-8 -*-
"""
@created on: 2/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import pandas as pd
import random
import h5py
import numpy as np


def subject_level_splitting(file):
    data = pd.read_csv(file)

    subject_groups = data.groupby(by='SUB_ID')
    subject_groups = [df for _, df in subject_groups]
    train_len = int(len(subject_groups) * 0.8)
    random.shuffle(subject_groups)

    train = pd.concat(subject_groups[:train_len]).reset_index(drop=True)
    test = pd.concat(subject_groups[train_len:]).reset_index(drop=True)

    train.to_csv('/Users/badgod/badgod_documents/Alco_audio/train.csv', index=False)
    test.to_csv('/Users/badgod/badgod_documents/Alco_audio/test.csv', index=False)


def read_h5py(filename, dataset_name='data'):
    print("Reading data from file ", filename)
    h5f = h5py.File(filename, 'r')
    data = np.array(h5f[dataset_name])
    h5f.close()

    return data


def save_h5py(data, filename, dataset_name='data'):
    print('Saving data in ', filename)
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(dataset_name, data=data)
    h5f.close()


def save_npy(data, filename):
    print('Saving data in ', filename)
    np.save(filename, data)


def read_npy(filename):
    print("Reading data from file ", filename)
    return np.load(filename, allow_pickle=True)


def save_csv(data, columns, filename):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)


# subject_level_splitting("/Users/badgod/badgod_documents/Alco_audio/alco_audio_data.csv")
