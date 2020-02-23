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
from pydub import AudioSegment
from pydub.utils import make_chunks


def melspectrogram_features(audio):
    # audio, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=10)
    mel = librosa.feature.melspectrogram(y=audio, n_mels=100)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel_norm = np.mean(logmel, axis=0)
    return logmel_norm


def split_audio_into_equal_chunks(file, milliseconds):
    # audio, sample_rate = librosa.load(
    #         "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/128bpm/AM_HiTeeb[A]_128D.wav",
    #         res_type='kaiser_fast', duration=10)
    # print(sample_rate)
    # print(audio.shape)
    # print(7.5 * sample_rate)
    # piece = audio[:int(7.5 * sample_rate)]
    # print("piece", piece.shape)
    # exit()
    # chunks = librosa.util.frame(audio, frame_length=1000, axis=0, hop_length=200)
    # print(len(chunks))
    audio, sample_rate = librosa.load(
            "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/128bpm/AM_HiTeeb[A]_128D.wav",
            res_type='kaiser_fast', duration=10)
    return [audio]


def preprocess_data(base_path, files, labels):
    data, out_labels = [], []
    for file, label in zip(files, labels):
        chunks = split_audio_into_equal_chunks(base_path + '/' + file, 10000)
        data.extend([melspectrogram_features(chunk) for chunk in chunks])
        out_labels.extend([float(label) for _ in range(len(chunks))])
    return data, out_labels

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
#     logmel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
#     S_dB = librosa.power_to_db(logmel, ref=np.max)
#     print(logmel.shape)
#     S_dB = np.mean(S_dB.T, axis=0)
#     print(S_dB.shape)
#
#     plt.figure(figsize=(12, 4))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time')
#     plt.show()
