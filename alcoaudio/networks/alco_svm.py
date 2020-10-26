# -*- coding: utf-8 -*-
"""
@created on: 10/26/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import numpy as np
from sklearn import svm

_min, _max = np.inf, -np.inf


def norm(data, train):
    if train:
        for x in data:
            _min = min(np.min(x), _min)
            _max = max(np.max(x), _max)

    data = (data - _min) / (_max - _min)
    return data

def metrics(true, pred):



filepath = '/Users/badgod/badgod_documents/Projects/Alco_audio/small_data/40_mels/'
train_jitter, train_label = 'train_challenge_with_shimmer_jitter.npy', 'train_challenge_with_d1_mel_labels.npy'
dev_jitter, dev_label = 'dev_challenge_with_shimmer_jitter.npy', 'dev_challenge_with_d1_mel_labels.npy'
test_jitter, test_label = 'test_challenge_with_shimmer_jitter.npy', 'test_challenge_with_d1_mel_labels.npy'

train_jitter, train_label = norm(np.load(filepath + train_jitter), train=True), np.load(train_label)
dev_jitter, dev_label = norm(np.load(filepath + dev_jitter), train=False), np.load(dev_label)
test_jitter, test_label = norm(np.load(filepath + test_jitter), train=False), np.load(test_label)

clf = svm.SVC()
clf.fit(train_jitter, train_label)

# predict
train_pred_labels = clf.predict(train_jitter)
dev_pred_labels = clf.predict(dev_jitter)
test_pred_labels = clf.predict(test_jitter)



