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
import pandas as pd
from sklearn import svm
from sklearn.metrics import recall_score, precision_recall_fscore_support


def norm(data, train, min_val, max_val):
    # if train:
    #     for x in data:
    #         min_val = min(np.min(x), min_val)
    #         max_val = max(np.max(x), max_val)
    #
    # data = (data - min_val) / (max_val - min_val)
    if train:
        return data, min_val, max_val
    else:
        return data


def metrics(true, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
    accuracy = np.sum(pred == true) / float(len(pred))
    uar = recall_score(true, pred, average='macro')
    return {"accuracy": accuracy, "uar": uar, "precision": precision, "recall": recall, "f1": f1}


if __name__ == '__main__':
    filepath = '/Users/badgod/badgod_documents/Projects/Alco_audio/small_data/40_mels/'
    train_jitter, train_label = 'train_challenge_with_opensmile.csv', 'train_challenge_with_d1_mel_labels.npy'
    dev_jitter, dev_label = 'dev_challenge_with_opensmile.csv', 'dev_challenge_with_d1_mel_labels.npy'
    test_jitter, test_label = 'test_challenge_with_opensmile.csv', 'test_challenge_with_d1_mel_labels.npy'

    min_val, max_val = np.inf, -np.inf
    train_jitter, min_val, max_val = norm(pd.read_csv(filepath + train_jitter).values, train=True, min_val=min_val,
                                          max_val=max_val)
    train_label = np.load(filepath + train_label)
    dev_jitter, dev_label = norm(pd.read_csv(filepath + dev_jitter).values, train=False, min_val=min_val,
                                 max_val=max_val), np.load(filepath + dev_label)
    test_jitter, test_label = norm(pd.read_csv(filepath + test_jitter).values, train=False, min_val=min_val,
                                   max_val=max_val), np.load(filepath + test_label)

    clf = svm.NuSVC(gamma='auto')
    clf.fit(train_jitter, train_label)

    # predict
    train_pred_labels = clf.predict(train_jitter)
    dev_pred_labels = clf.predict(dev_jitter)
    test_pred_labels = clf.predict(test_jitter)

    print(metrics(train_label, train_pred_labels))
    print(metrics(dev_label, dev_pred_labels))
    print(metrics(test_label, test_pred_labels))
