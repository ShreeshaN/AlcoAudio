# -*- coding: utf-8 -*-
"""
@created on: 2/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from torch import tensor
import torch
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np


def accuracy_fn(preds, labels, threshold):
    # todo: UAR implementation is wrong. Tweak it once the model is ready
    predictions = torch.where(preds > tensor(threshold), tensor(1), tensor(0))
    accuracy = torch.sum(predictions == labels) / float(len(labels))
    uar = recall_score(labels, predictions.numpy(), average='macro')
    return accuracy, uar


def log_summary(writer, global_step, accuracy, loss, uar, lr, type):
    writer.add_scalar(f'{type}/Accuracy', accuracy, global_step)
    writer.add_scalar(f'{type}/Loss', loss, global_step)
    writer.add_scalar(f'{type}/UAR', uar, global_step)
    writer.add_scalar(f'{type}/LR', lr, global_step)
    writer.flush()


def log_conf_matrix(writer, global_step, predictions_dict, type):
    writer.add_scalars(f'{type}/Predictions Average', {x: np.mean(predictions_dict[x]) for x in predictions_dict},
                       global_step)
    writer.add_scalars(f'{type}/Predictions Count', {x: len(predictions_dict[x]) for x in predictions_dict},
                       global_step)
    writer.flush()


def normalize_image(image):
    # return (image - image.mean())/image.std()
    return (image - image.min()) / (image.max() - image.min())


def custom_confusion_matrix(preds, target, threshold=0.5):
    preds = torch.where(preds > tensor(threshold), tensor(1), tensor(0))
    TP = []
    FP = []
    TN = []
    FN = []

    for i in range(len(preds)):
        if target[i] == preds[i] == 1:
            TP.append(preds[i])
        if preds[i] == 1 and target[i] != preds[i]:
            FP.append(preds[i])
        if target[i] == preds[i] == 0:
            TN.append(preds[i])
        if preds[i] == 0 and target[i] != preds[i]:
            FN.append(preds[i])

    return TP, FP, TN, FN
