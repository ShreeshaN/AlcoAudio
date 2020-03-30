# -*- coding: utf-8 -*-
"""
@created on: 3/30/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import random
from alcoaudio.datagen.nb_SparseImageWarp import sparse_image_warp


def librosaSpectro_to_torchTensor(librosa_spectro):
    mel_spectro = torch.from_numpy(librosa_spectro)
    mel_spectro = torch.unsqueeze(mel_spectro, 0)
    return mel_spectro


def tensor_to_img(spectrogram, sr, method):
    spectrogram = torch.squeeze(spectrogram, 0)
    spect = spectrogram.numpy()
    plt.figure()
    S_dB = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.title(method)
    plt.show()


def plot_librosa_spectro(spectrogram, sr):
    plt.figure()
    S_dB = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.title('Mel-frequency librosa spectrogram')
    plt.show()


def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned


# drHalf_audio_path = '/Users/pratcr7/Desktop/Alco_aug/Data/BL10_0.00063/0651066015_h_00.wav'
# raw_audio_path = '/Users/badgod/badgod_documents/Alco_audio/data/ALC/DATA/audio_4.wav'
#
# y3, sr3 = librosa.load(raw_audio_path, mono=False)
#
# mel_spectro_librosa = librosa.feature.melspectrogram(y=y3, sr=sr3, S=None, n_fft=1024, hop_length=256, n_mels=128)
# print(mel_spectro_librosa.shape)
# # plot_librosa_spectro(mel_spectro_librosa, sr3)
#
# mel_spectro = librosaSpectro_to_torchTensor(mel_spectro_librosa)
#
# # timeWarp_spectro = time_warp(mel_spectro)
# # tensor_to_img(timeWarp_spectro, sr=sr3, method='Time Warp')
#
# timeMask_spectro = time_mask(mel_spectro)
# print(timeMask_spectro[0].numpy().shape)
# exit()
# tensor_to_img(timeMask_spectro, sr=sr3, method='Time Mask')
#
# freqMask_spectro = freq_mask(mel_spectro)
# tensor_to_img(freqMask_spectro, sr=sr3, method='Frequency Mask')
