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


subject_level_splitting("/Users/badgod/badgod_documents/Alco_audio/alco_audio_data.csv")
