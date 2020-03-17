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
from joblib import Parallel, delayed


def mfcc_features(audio, normalise=False):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=40)
    if normalise:
        mfcc_norm = np.mean(mfcc.T, axis=0)
        return mfcc_norm
    else:
        return mfcc


def mel_filters(audio, normalise=False):
    logmel = librosa.feature.melspectrogram(y=audio, n_mels=40)
    if normalise:
        return librosa.power_to_db(np.mean(logmel.T), ref=np.max)
    else:
        return librosa.power_to_db(logmel, ref=np.max)


def mel_filters_with_spectrogram(audio, filename, normalise=False):
    plt.figure(figsize=(3, 2))
    logmel = librosa.feature.melspectrogram(y=audio, n_mels=128)
    logmel = librosa.power_to_db(logmel, ref=np.max)
    librosa.display.specshow(logmel)
    plt.savefig(filename)
    plt.close()


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
    # if os.path.exists(base_path + file):
    #     audio, sr = librosa.load(base_path + file)
    #     chunks = get_audio_list(audio, sr=sampling_rate, cut_length=sample_size_in_seconds, overlap=overlap)
    #     # [(data.append(mfcc_features(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
    #     # [(data.extend(mel_filters(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
    #     for chunk in chunks:
    #         features = mel_filters(chunk, normalise)
    #         data.append(features)
    #         out_labels.append(float(label))
    # return data, out_labels
    data, out_labels = [], []
    if os.path.exists(base_path + file):
        audio, sr = librosa.load(base_path + file)
        chunks = get_audio_list(audio, sr=sampling_rate, cut_length=sample_size_in_seconds, overlap=overlap)
        # [(data.append(mfcc_features(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
        # [(data.extend(mel_filters(chunk, normalise)), out_labels.append(float(label))) for chunk in chunks]
        for chunk in chunks:
            features = mfcc_features(chunk, normalise)
            if 345 in features.shape:
                data.append(features)
                out_labels.append(float(label))
            else:
                # for logging
                print('FIle with issue ', file)
    return data, out_labels


def preprocess_data(base_path, files, labels, normalise, sample_size_in_seconds, sampling_rate, overlap):
    data, out_labels = [], []
    aggregated_data = Parallel(n_jobs=4, backend='multiprocessing')(
            delayed(read_audio_n_process)(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap,
                                          normalise) for file, label in
            tqdm(zip(files, labels), total=len(labels)))

    for per_file_data in aggregated_data:
        # per_file_data[1] are labels for the audio file.
        # Might be an array as one audio file can be split into many pieces based on sample_size_in_seconds parameter
        for i, label in enumerate(per_file_data[1]):
            # per_file_data[0] is array of audio samples based on sample_size_in_seconds parameter
            if 345 in per_file_data[0][i].shape:  # Temp fix
                data.append(per_file_data[0][i])
                out_labels.append(label)
    return data, out_labels



# for images
def read_audio_n_save_spectrograms(file, label, base_path, image_save_path, sampling_rate, sample_size_in_seconds,
                                   overlap, normalise):
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
    data.sort()
    if os.path.exists(base_path + file):
        audio, sr = librosa.load(base_path + file)
        chunks = get_audio_list(audio, sr=sampling_rate, cut_length=sample_size_in_seconds, overlap=overlap)
        for i, chunk in enumerate(chunks):
            filename = image_save_path + file.split("/")[-1] + "_" + str(i) + "_label_" + str(label) + '.jpg'
            # mel_filters_with_spectrogram(chunk, filename, normalise)
            mfcc_features(chunk, normalise)
            data.append(filename)
            out_labels.append(float(label))
    return data, out_labels


def preprocess_data_images(base_path, image_save_path, files, labels, normalise, sample_size_in_seconds, sampling_rate,
                    overlap):
    data, out_labels = [], []
    aggregated_data = Parallel(n_jobs=4, backend='multiprocessing')(
            delayed(read_audio_n_save_spectrograms)(file, label, base_path, image_save_path, sampling_rate,
                                                    sample_size_in_seconds,
                                                    overlap,
                                                    normalise) for file, label in
            tqdm(zip(files, labels), total=len(labels)))
    for x in aggregated_data:
        data.extend(x[0])
        out_labels.extend(x[1])
    return data, out_labels


# def preprocess_data(base_path, files, labels, normalise, sample_size_in_seconds, sampling_rate, overlap):
#     data, out_labels = [], []
#     aggregated_data = Parallel(n_jobs=4, backend='threading')(
#             delayed(read_audio_n_process)(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap,
#                                           normalise) for file, label in tqdm(zip(files, labels), total=len(labels)))
#
#     for per_file_data in aggregated_data:
#         # per_file_data[1] are labels for the audio file.
#         # Might be an array as one audio file can be split into many pieces based on sample_size_in_seconds parameter
#         for i, label in enumerate(per_file_data[1]):
#             # per_file_data[0] is array of audio samples based on sample_size_in_seconds parameter
#             if 345 in per_file_data[0][i].shape:  # Temp fix
#                 data.append(per_file_data[0][i])
#                 out_labels.append(label)
#     return data, out_labels


# file = '/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/AM_HiTeeb[A]_132D.wav'
# note, sr = librosa.load(file)
# print(note.shape)
# list_y = get_audio_list(note)
# print([librosa.output.write_wav(
#         "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/" + str(i) + ".wav", x, 22050) for
#     i, x
#     in enumerate(list_y)])

def mfcc():
    # this is mel filters + dct
    file_name = '/Users/badgod/Downloads/AC_12Str85F-01.mp3'
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    print("audio, sample_rate", audio.shape, sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    print(mfccs.shape)
    print(np.max(mfccs), np.min(mfccs))
    # exit()
    mfccsscaled = np.mean(mfccs.T, axis=0)
    print(mfccsscaled.shape)

    plt.figure(figsize=(12, 4))
    # plt.plot(audio)
    # plt.plot(mfccsscaled)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.show()


#

# mfcc()

# def mel_filters_x():
#     file_name = '/Users/badgod/Downloads/AC_12Str85F-01.mp3'
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#
#     print("audio, sample_rate", audio.shape, sample_rate)
#     logmel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
#     print(logmel.shape)
#     print(np.min(logmel), np.max(logmel))
#     S_dB = librosa.power_to_db(logmel, ref=np.max)
#     print(np.min(S_dB), np.max(S_dB))
#     print(S_dB[0].shape)
#     print(S_dB.shape)
#     print(S_dB.mean())
#     # S_dB = S_dB / 255
#     print(S_dB.mean())
#     # exit()
#     # S_dB = np.mean(S_dB.T, axis=0)
#     # print(S_dB.shape)
#
#     plt.figure(figsize=(4, 3))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     # librosa.display.specshow(logmel, sr=sample_rate, x_axis='time')
#     librosa.display.specshow(S_dB, sr=sample_rate)
#     plt.xlabel('Time')
#     plt.ylabel('Mels')
#     plt.savefig("test_40mels.jpg")
#
#     # plt.plot(S_dB)
#     plt.show()
#
#     plt.close()

# mel_filters_x()
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

# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# a = mpimg.imread("/Users/badgod/badgod_documents/github/AlcoAudio/alcoaudio/datagen/test.jpg")
# b = cv2.imread("/Users/badgod/badgod_documents/github/AlcoAudio/alcoaudio/datagen/test.jpg")
# b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
# # b = (b-b.mean())/b.std()
# b = (b - b.min()) / (b.max() - b.min())
# print(a.shape, b.shape)
# print(a[:, :, 0].mean(), a[:, :, 1].mean(), a[:, :, 2].mean())
# print(b[:, :, 0].mean(), b[:, :, 1].mean(), b[:, :, 2].mean())
# plt.imshow(b)
# plt.show()
