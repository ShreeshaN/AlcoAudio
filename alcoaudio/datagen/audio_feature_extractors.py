# -*- coding: utf-8 -*-
"""
@created on: 2/17/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import math
import os
import subprocess

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysptk
import torch
import wavio
from joblib import Parallel, delayed
from pyannote.audio.utils.signal import Binarize
from pyts.image import GramianAngularField
from tqdm import tqdm


def mfcc_features(audio, sampling_rate, normalise=False):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=40, sr=sampling_rate)
    if normalise:
        mfcc_norm = np.mean(mfcc.T, axis=0)
        return mfcc_norm
    else:
        return mfcc


def mel_filters(audio, sampling_rate, normalise=False):
    mel_spec = librosa.feature.melspectrogram(y=audio, n_mels=40, sr=sampling_rate)
    if normalise:
        return np.mean(librosa.power_to_db(mel_spec, ref=np.max).T)
    else:
        return librosa.power_to_db(mel_spec, ref=np.max)
        #


def mel_filters_with_spectrogram(audio, sampling_rate, filename, normalise=False):
    plt.figure(figsize=(3, 2))
    logmel = librosa.feature.melspectrogram(y=audio, n_mels=40, sr=sampling_rate)
    logmel = librosa.power_to_db(logmel, ref=np.max)
    librosa.display.specshow(logmel)
    plt.savefig(filename)
    plt.close()


def gaf(audio):
    gasf = GramianAngularField(image_size=100, method='summation')
    X_gasf = gasf.fit_transform(audio.reshape((-1, len(audio))))
    _, x, y = X_gasf.shape
    return np.float32(X_gasf.reshape((x, y)))


def cut_audio(audio, sampling_rate, sample_size_in_seconds, overlap):
    """
    Method to split a audio signal into pieces based on `sample_size_in_seconds` and `overlap` parameters
    :param audio: The main audio signal to be split
    :param sampling_rate: The rate at which audio is sampled
    :param sample_size_in_seconds: number of seconds in each split
    :param overlap: in seconds, how much of overlap is required within splits
    :return: List of splits
    """
    if overlap >= sample_size_in_seconds:
        raise Exception("Please maintain this condition: sample_size_in_seconds > overlap")

    def add_to_audio_list(y):
        if len(y) / sampling_rate < sample_size_in_seconds:
            raise Exception(
                    f'Length of audio lesser than `sampling size in seconds` - {len(y) / sampling_rate} seconds, required {sample_size_in_seconds} seconds')
        y = y[:required_length]
        audio_list.append(y)

    audio_list = []
    required_length = sample_size_in_seconds * sampling_rate
    audio_in_seconds = len(audio) // sampling_rate

    # Check if the main audio file is larger than the required number of seconds
    if audio_in_seconds >= sample_size_in_seconds:
        start = 0
        end = sample_size_in_seconds
        left_out = None

        # Until highest multiple of sample_size_in_seconds is reached, ofcourse, wrt audio_in_seconds, run this loop
        while end <= audio_in_seconds:
            index_at_start, index_at_end = start * sampling_rate, end * sampling_rate
            one_audio_sample = audio[index_at_start:index_at_end]
            add_to_audio_list(one_audio_sample)
            left_out = audio_in_seconds - end
            start = (start - overlap) + sample_size_in_seconds
            end = (end - overlap) + sample_size_in_seconds

        # Whatever is left out after the iteration, just include that to the final list.
        # Eg: if 3 seconds is left out and sample_size_in_seconds is 5 seconds, then cut the last 5 seconds of the audio
        # and append to final list.
        if left_out > 0:
            one_audio_sample = audio[-sample_size_in_seconds * sampling_rate:]
            add_to_audio_list(one_audio_sample)
    # Else, just repeat the required number of seconds at the end. The repeated audio is taken from the start
    else:
        less_by = sample_size_in_seconds - audio_in_seconds
        excess_needed = less_by * sampling_rate
        one_audio_sample = np.append(audio, audio[-excess_needed:])

        # This condition is for samples which are too small and need to be repeated
        # multiple times to satisfy the `sample_size_in_seconds` parameter
        while len(one_audio_sample) < (sampling_rate * sample_size_in_seconds):
            one_audio_sample = np.hstack((one_audio_sample, one_audio_sample))
        add_to_audio_list(one_audio_sample)
    return audio_list


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for e, mean in enumerate(y_mean):
        # print('Mean - ', e, int(e/rate) ,mean) if e%500==0 else None
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def remove_silent_parts(filepath, sr, model):
    audio, sr = librosa.load(filepath, sr=sr)
    test_file = {'uri': filepath.split('/')[-1], 'audio': filepath}

    # obtain raw SAD scores (as `pyannote.core.SlidingWindowFeature` instance)
    sad_scores = model(test_file)

    # binarize raw SAD scores
    # NOTE: both onset/offset values were tuned on AMI dataset.
    # you might need to use different values for better results.
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True,
                        min_duration_off=0.1, min_duration_on=0.1)

    # speech regions (as `pyannote.core.Timeline` instance)
    speech = binarize.apply(sad_scores, dimension=1)

    audio_pieces = []
    for segment in speech:
        segment = list(segment)
        audio_pieces.extend(audio[int(segment[0] * sr):int(segment[1] * sr)])
    return np.array(audio_pieces)


def get_shimmer_jitter_from_opensmile(audio, index, sr):
    wavio.write(f'temp_{str(index)}.wav', audio, sr, sampwidth=3)
    subprocess.call(
            ["SMILExtract", "-C", os.environ['OPENSMILE_CONFIG_DIR'] + "/IS10_paraling.conf", "-I",
             f"temp_{str(index)}.wav", "-O",
             f"temp_{str(index)}.arff"])
    # Read file and extract shimmer and jitter features from the generated arff file
    file = open(f"temp_{str(index)}", "r")
    data = file.readlines()

    # First 3 values are title, empty line and name | Last 5 values are numeric data,
    # and bunch of empty lines and unwanted text
    headers = data[3:-5]

    # Last line of data is where the actual numeric data is. It is in comma separated string format. After splitting,
    # remove the first value which is name and the last value which is class
    numeric_data = data[-1].split(',')[1:-1]

    assert len(headers) == len(numeric_data), "Features generated from opensmile are not matching with its headers"

    # data_needed = {x.strip(): float(numeric_data[e]) for e, x in enumerate(headers) if 'jitter' in x or 'shimmer' in x}
    data_needed = [float(numeric_data[e]) for e, x in enumerate(headers) if 'jitter' in x or 'shimmer' in x]
    return data_needed


def read_audio_n_process(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise, method):
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
    filepath = base_path + file
    if os.path.exists(filepath):
        audio, sr = librosa.load(filepath, sr=sampling_rate)
        # mask = envelope(audio, sr, 0.0005)
        # audio = audio[mask]
        sr = sampling_rate
        # audio = remove_silent_parts(filepath, sr=sampling_rate)
        chunks = cut_audio(audio, sampling_rate=sr, sample_size_in_seconds=sample_size_in_seconds,
                           overlap=overlap)
        for e, chunk in enumerate(chunks):
            zero_crossing = librosa.feature.zero_crossing_rate(chunk)
            f0 = pysptk.swipe(chunk.astype(np.float64), fs=sr, hopsize=510, min=60, max=240, otype="f0").reshape(1, -1)
            pitch = pysptk.swipe(chunk.astype(np.float64), fs=sr, hopsize=510, min=60, max=240, otype="pitch").reshape(
                    1, -1)
            f0_pitch_multiplier = 1
            features = mel_filters(chunk, sr, normalise)
            f0 = np.reshape(f0[:, :features.shape[1] * f0_pitch_multiplier], newshape=(f0_pitch_multiplier, -1))
            pitch = np.reshape(pitch[:, :features.shape[1] * f0_pitch_multiplier], newshape=(f0_pitch_multiplier, -1))
            shimmer_jitter = get_shimmer_jitter_from_opensmile(chunk, e, sr)
            shimmer_jitter = np.tile(shimmer_jitter, math.floor(len(features) / len(shimmer_jitter)))[
                             :len(shimmer_jitter)]  # Repeating the values to match the features length of filterbanks
            if method == 'fbank':
                features = np.concatenate((features, zero_crossing, f0, pitch, shimmer_jitter), axis=0)
            elif method == 'mfcc':
                features = mfcc_features(chunk, sr, normalise)
            elif method == 'gaf':
                features = gaf(chunk)
            elif method == 'raw':
                features = chunk
            else:
                raise Exception(
                        'Specify a method to use for pre processing raw audio signal. Available options - {fbank, mfcc, gaf, raw}')
            data.append(features)
            out_labels.append(float(label))
    else:
        print('File not found ', filepath)
    return data, out_labels


def preprocess_data(base_path, files, labels, normalise, sample_size_in_seconds, sampling_rate, overlap, method):
    data, out_labels = [], []
    aggregated_data = Parallel(n_jobs=8, backend='multiprocessing')(
            delayed(read_audio_n_process)(file, label, base_path, sampling_rate, sample_size_in_seconds, overlap,
                                          normalise, method) for file, label in
            tqdm(zip(files, labels), total=len(labels)))

    for per_file_data in aggregated_data:
        # per_file_data[1] are labels for the audio file.
        # Might be an array as one audio file can be split into many pieces based on sample_size_in_seconds parameter
        for i, label in enumerate(per_file_data[1]):
            # per_file_data[0] is array of audio samples based on sample_size_in_seconds parameter
            data.append(per_file_data[0][i])
            out_labels.append(label)
    return data, out_labels


def remove_silent_parts_from_audio(base_path, files, sampling_rate):
    sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
    for file in tqdm(files):
        filepath = base_path + file
        if os.path.exists(filepath):
            sr = sampling_rate
            audio = remove_silent_parts(filepath, sr=sampling_rate, model=sad)
            path_to_create = base_path + 'removed/' + file
            os.makedirs('/'.join(path_to_create.split('/')[:-1]), exist_ok=True)
            librosa.output.write_wav(path_to_create, audio, sr)
        else:
            print('File not found ', filepath)


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
        chunks = cut_audio(audio, sampling_rate=sampling_rate, sample_size_in_seconds=sample_size_in_seconds,
                           overlap=overlap)
        for i, chunk in enumerate(chunks):
            filename = image_save_path + file.split("/")[-1] + "_" + str(i) + "_label_" + str(label) + '.jpg'
            mel_filters_with_spectrogram(chunk, sampling_rate, filename, normalise)
            # mfcc_features(chunk, normalise)
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

    for per_file_data in aggregated_data:
        # per_file_data[1] are labels for the audio file.
        # Might be an array as one audio file can be split into many pieces based on sample_size_in_seconds parameter
        for i, label in enumerate(per_file_data[1]):
            # per_file_data[0] is array of audio samples based on sample_size_in_seconds parameter
            data.append(per_file_data[0][i])
            out_labels.append(label)
    return data, out_labels

############################## TESTING ##############################
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
#     print(np.max(mfccs), np.min(mfccs))
#     # exit()
#     mfccsscaled = np.mean(mfccs.T, axis=0)
#     print(mfccsscaled.shape)
#
#     plt.figure(figsize=(12, 4))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
#     plt.show()


#

# mfcc()

# def mel_filters_x():
#     file_name = '/Users/badgod/badgod_documents/Projects/Alco_audio/data/ALC/DATA/audio_2.wav'
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#
#     print("audio, sample_rate", audio.shape, sample_rate)
#     # plt.plot(range(len(audio)), audio)
#     # plt.savefig('/Users/badgod/badgod_documents/Projects/Alco_audio/raw_signal.jpg')
#     # plt.show()
#     # exit()
#     logmel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
#     print("melspectrogram ", logmel.shape)
#     # exit()
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
#     # plt.figure(figsize=(12, 8))
#     # plt.plot(audio)
#     # plt.plot(mfccsscaled)
#     # librosa.display.specshow(logmel, sr=sample_rate, x_axis='time')
#     librosa.display.specshow(S_dB, sr=sample_rate)
#     plt.xlabel('Time')
#     plt.ylabel('Mels')
#     plt.savefig("/Users/badgod/badgod_documents/Projects/Alco_audio/test_40mels.jpg")
#
#     # plt.plot(S_dB)
#     plt.show()
#
#     plt.close()
#
#
# mel_filters_x()
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = np.load("/Users/badgod/badgod_documents/Alco_audio/small_data/40_mels/train_challenge_data.npy",
#                allow_pickle=True)
# print(data.shape)
#
# data_means = np.array([x.mean() for x in data])
#
# plt.hist(data_means)
# plt.show()
############################## TESTING ##############################
