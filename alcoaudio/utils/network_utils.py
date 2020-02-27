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


def accuracy_fn(predictions, labels, threshold):
    # todo: UAR implementation is wrong. Tweak it once the model is ready
    predictions = torch.where(predictions > tensor(threshold), tensor(1), tensor(0))
    accuracy = torch.sum(predictions == tensor(labels)) / float(len(labels))
    uar = recall_score(labels, predictions.numpy(), average='macro')
    return accuracy, uar
