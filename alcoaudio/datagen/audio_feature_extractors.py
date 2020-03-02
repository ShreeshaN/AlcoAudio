# -*- coding: utf-8 -*-
"""
@created on: 2/17/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time


def mfcc_features(audio, normalise=False):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=40)
    if normalise:
        mfcc_norm = np.mean(mfcc.T, axis=0)
        return mfcc_norm
    else:
        return mfcc


def mel_filters(audio, normalise=False):
    logmel = librosa.feature.melspectrogram(y=audio, n_mels=128)
    if normalise:
        return librosa.power_to_db(np.mean(logmel.T), ref=np.max)
    else:
        return librosa.power_to_db(logmel, ref=np.max)


def get_audio_list(audio, sr=22050, cut_length=10, overlap=1):
    y, trim_idx = librosa.effects.trim(audio)  #
    len_sample = cut_length * sr  # array length of sample
    len_ol = overlap * sr  # overlaplength(array)
    y_mat = []  # initiate list
    i = 1  # iterator

    if (trim_idx[1] < len_sample):  # check if voice note is too small
        no_of_times_to_replicate = int(len_sample / trim_idx[1]) + 1  # Calculating the replication factor if audio
        return [np.tile(audio, no_of_times_to_replicate)[:len_sample]]  # Trimming the extra audio that is extra
    else:
        while (i * len_sample - (i - 1) * len_ol <= trim_idx[1]):
            trim_y = y[(i - 1) * len_sample - (i - 1) * len_ol: i * len_sample - (i - 1) * len_ol]  # trim voice notes
            y_mat.append(trim_y)
            i = i + 1
        if ((i - 1) * len_sample - (i - 2) * len_ol < trim_idx[1]):
            trim_y = y[trim_idx[1] - len_sample:trim_idx[1]]  # addition of remainder voice note
            y_mat.append(trim_y)

        return y_mat  # return list


# def preprocess_data(base_path, files, labels, normalise, sample_size_in_seconds, sampling_rate, overlap):
#     data, out_labels = [], []
#     for file, label in zip(files, labels):
#         if not os.path.exists(base_path + file):
#             continue
#         audio, sr = librosa.load(base_path + file)
#         chunks = get_audio_list(audio, sr=sampling_rate, cut_length=sample_size_in_seconds, overlap=overlap)
#         data.extend([mfcc_features(chunk, normalise) for chunk in chunks])
#         out_labels.extend([float(label) for _ in range(len(chunks))])
#     return data, out_labels


from joblib import Parallel, delayed


def read_audio_n_process(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise):
    """
    This method is called by the preprocess data method
    :param file:
    :param label:
    :param base_path:
    :param sampling_rate:
    :param sample_size_in_seconds:
    :param overlap:
    :param normalise:
    :return:
    """
    data, out_labels = [], []
    audio, sr = librosa.load(base_path + file)
    chunks = get_audio_list(audio, sr=sampling_rate, cut_length=sample_size_in_seconds, overlap=overlap)
    # [(data.append(mfcc_features(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
    [(data.append(mel_filters(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
    return [data, out_labels]


def preprocess_data(base_path, files, labels, normalise, sample_size_in_seconds, sampling_rate, overlap):
    data, out_labels = [], []
    results = Parallel(n_jobs=4, backend='threading')(
            delayed(read_audio_n_process)(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap,
                                          normalise) for
            file, label in
            tqdm(zip(files, labels), total=len(labels)))
    [(data.extend(x[0]), out_labels.extend(x[1])) for x in results]
    # idx = [i for i, x in enumerate(data) if len(x) == 345]
    #
    # # TEMP FIX
    # if len(idx) == len(data):
    #     return data, out_labels
    # else:
    #     data, out_labels = np.array(data)[idx], np.array(out_labels)[idx]
    #     data = np.array([x.reshape(345) for x in data])
    #     data = data.reshape(len(idx), 345)
    #     return data, out_labels
    return data, out_labels


# file = '/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/AM_HiTeeb[A]_132D.wav'
# note, sr = librosa.load(file)
# print(note.shape)
# list_y = get_audio_list(note)
# print([librosa.output.write_wav(
#         "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/" + str(i) + ".wav", x, 22050) for
#     i, x
#     in enumerate(list_y)])

# def mfcc():
#     # this is mel filters + dct
#     file_name = '/Users/badgod/Downloads/AC_12Str85F-01.mp3'
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#
#     print("audio, sample_rate", audio.shape, sample_rate)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     print(mfccs.shape)
#     mfccsscaled = np.mean(mfccs.T, axis=0)
#     print(mfccsscaled.shape)
#
#     plt.figure(figsize=(12, 4))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
#     plt.show()
#
# def mel_filters():
#     file_name = '/Users/badgod/Downloads/AC_12Str85F-01.mp3'
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#
#     print("audio, sample_rate", audio.shape, sample_rate)
#     logmel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
#     print(logmel.shape)
#     S_dB = librosa.power_to_db(logmel, ref=np.max)
#     print(S_dB.shape)
#     # S_dB = np.mean(S_dB.T, axis=0)
#     # print(S_dB.shape)
#
#     plt.figure(figsize=(12, 4))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     # librosa.display.specshow(logmel, sr=sample_rate, x_axis='time')
#     a = librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time')
#     print(a)
#     # plt.plot(S_dB)
#     plt.show()


# mel_filters()
# def split_audio_into_equal_chunks(file, milliseconds):
#     # audio, sample_rate = librosa.load(
#     #         "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/128bpm/AM_HiTeeb[A]_128D.wav",
#     #         res_type='kaiser_fast', duration=10)
#     # print(sample_rate)
#     # print(audio.shape)
#     # print(7.5 * sample_rate)
#     # piece = audio[:int(7.5 * sample_rate)]
#     # print("piece", piece.shape)
#     # exit()
#     # chunks = librosa.util.frame(audio, frame_length=1000, axis=0, hop_length=200)
#     # print(len(chunks))
#     audio, sample_rate = librosa.load(
#             "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/128bpm/AM_HiTeeb[A]_128D.wav",
#             res_type='kaiser_fast', duration=10)
#     return [audio]
