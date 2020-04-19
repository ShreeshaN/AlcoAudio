# -*- coding: utf-8 -*-
"""
@created on: 4/18/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score


def norm(data):
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val)
    return data.mean(axis=1)


def uar(y, yhat):
    return recall_score(y, yhat, average='macro')


train_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_data.npy")
train_data = norm(train_data)
train_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_labels.npy")
inliers_ids, outliers_ids = [i for i, x in enumerate(train_labels) if x == 0], [i for i, x in enumerate(train_labels) if
                                                                                x == 1]
train_inliers, train_outliers = train_data[inliers_ids], train_data[outliers_ids]
dev_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_data.npy")
dev_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_labels.npy")
dev_data = norm(dev_data)

test_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_data.npy")
test_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy")
test_data = norm(test_data)

outlier_prop = len(train_outliers) / len(train_inliers)
svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma='auto')
svm.fit(train_inliers)
y_pred = svm.predict(test_data)

y_pred = [max(0, x) for x in y_pred]
print(uar(test_labels, y_pred))
