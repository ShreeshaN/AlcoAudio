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


def log_summary(writer, global_step, tr_accuracy, tr_loss, tr_uar, te_accuracy, te_loss, te_uar):
    writer.add_scalar('Train/Epoch Accuracy', tr_accuracy, global_step)
    writer.add_scalar('Train/Epoch Loss', tr_loss, global_step)
    writer.add_scalar('Train/Epoch UAR', tr_uar, global_step)
    writer.add_scalar('Test/Accuracy', te_accuracy, global_step)
    writer.add_scalar('Test/Loss', te_loss, global_step)
    writer.add_scalar('Test/UAR', te_uar, global_step)
    writer.flush()
