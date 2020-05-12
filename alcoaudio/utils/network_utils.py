# # -*- coding: utf-8 -*-
# """
# @created on: 2/23/20,
# @author: Shreesha N,
# @version: v0.0.1
# @system name: badgod
# Description:
#
# ..todo::
#
# """

from torch import tensor
import torch
from sklearn.metrics import recall_score
import numpy as np
import os


def to_tensor(x, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tensor(x).to(device=device).float()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            return x.detach().cpu().numpy()
        else:
            return x.cpu().numpy()
    else:
        return x


def accuracy_fn(preds, labels, threshold):
    # todo: UAR implementation is wrong. Tweak it once the model is ready
    predictions = torch.where(preds > to_tensor(threshold), to_tensor(1), to_tensor(0))
    accuracy = torch.sum(predictions == labels) / float(len(labels))
    uar = recall_score(to_numpy(labels), to_numpy(predictions), average='macro')
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


def write_to_npy(filename, **kwargs):
    state_dict = kwargs
    epoch = state_dict['epoch']
    state_dict.pop('epoch')

    # read
    if os.path.exists(filename):
        # append
        d = np.load(filename, allow_pickle=True)[0]
        d[epoch] = state_dict
        np.save(filename, [d])
    else:
        # create
        np.save(filename, [{epoch: state_dict}])


def normalize_image(image):
    # return (image - image.mean())/image.std()
    return (image - image.min()) / (image.max() - image.min())


def custom_confusion_matrix(predictions, target, threshold=0.5):
    preds = torch.where(predictions > to_tensor(threshold), to_tensor(1), to_tensor(0))
    TP = []
    FP = []
    TN = []
    FN = []

    for i in range(len(preds)):
        if target[i] == preds[i] == 1:
            TP.append(to_numpy(predictions[i]))
        if preds[i] == 1 and target[i] != preds[i]:
            FP.append(to_numpy(predictions[i]))
        if target[i] == preds[i] == 0:
            TN.append(to_numpy(predictions[i]))
        if preds[i] == 0 and target[i] != preds[i]:
            FN.append(to_numpy(predictions[i]))

    return TP, FP, TN, FN
