# -*- coding: utf-8 -*-
"""
@created on: 2/27/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import pandas as pd
import argparse
import json
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from alcoaudio.utils.class_utils import AttributeDict
from alcoaudio.datagen.audio_feature_extractors import preprocess_data, preprocess_data_images, preprocess_data_with_demogra
from alcoaudio.utils.data_utils import save_h5py, save_npy, save_csv


def parse():
    parser = argparse.ArgumentParser(description="alcoaudio_data_processor")
    parser.add_argument('--configs_file', type=str)
    args = parser.parse_args()
    return args


class DataProcessor:
    def __init__(self, args):
        self.base_path = args.audio_basepath
        self.train_data_file = args.train_data_file
        self.dev_data_file = args.dev_data_file
        self.test_data_file = args.test_data_file
        self.train_demog_file = args.train_demog_file
        self.dev_demog_file = args.dev_demog_file
        self.test_demog_file = args.test_demog_file
        self.normalise = args.normalise_while_creating
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_save_path = args.data_save_path
        self.image_save_path = args.image_data_save_path
        self.method = args.data_processing_method

    def process_audio_and_save_h5py(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file)
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data(self.base_path, df['WAV_PATH'].values, df['label'].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        save_h5py(data, self.data_save_path + '/' + filename_to_save + '_data.h5')
        save_h5py(labels, self.data_save_path + '/' + filename_to_save + '_labels.h5')

    def process_audio_and_save_npy(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file)
        print('Read a ')
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data(self.base_path, df['WAV_PATH'].values, df['label'].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        save_npy(data, self.data_save_path + '/' + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + '/' + filename_to_save + '_labels.npy')

    def process_audio_and_save_npy_challenge(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file, header=None, delimiter='\t')
        print('Number of audio files ', len(df))
        if shuffle:
            df = df.sample(frac=1)

        # Converting 'A' to 1 and 'N' to 0 - according to Challenge's binary decision.
        # Refer Challenge's Readme for further information
        df[1] = df[1].apply(lambda x: 1 if x == 'A' else 0)

        # Irregular use of extensions in data, so handling it here
        df[0] = df[0].apply(lambda x: x.replace('WAV', 'wav'))
        data, labels = preprocess_data(self.base_path, df[0].values, df[1].values,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap, self.method)
        print('Number of audio files after processing ', len(data))
        save_npy(data, self.data_save_path  + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + filename_to_save + '_labels.npy')

    def one_hot_Encoder(self):
        df_demog_train = pd.read_csv(self.train_demog_file, header=None)
        df_demog_dev = pd.read_csv(self.dev_demog_file, header=None)
        df_demog_test = pd.read_csv(self.test_demog_file, header=None)

        df_demog = pd.concat([df_demog_train,df_demog_dev,df_demog_test],axis=0)
        df_demog  = df_demog.reset_index(drop=True)

        df_demog = df_demog.drop(1,axis=1) #Dropping labels from demographics file
        df_demog_2B_enc = df_demog.iloc[:,2] #Separating filenames before encoding

        print('df_demog shape ', df_demog.shape)
        print('df_demog_2B_enc shape ', df_demog_2B_enc.shape)

        # demogra = df_demog_2B_enc.values  
        # #OneHotEncoding process for demographics
        # demogra = demogra.astype(str)
        # demogra = demogra.reshape(-1,1)
        # ohe = OneHotEncoder() 
        # ohe.fit(demogra)    
        # demogra_enc = ohe.transform(demogra)
        # demogra_enc = pd.DataFrame(demogra_enc.todense()) #Converting from csr_matrix to DF
        # demogra_enc = demogra_enc.drop(0, axis=1)  #Dropping Male column since binary Female column is sufficient for gender
        demogra_enc = df_demog_2B_enc.apply(lambda x: 1 if x == 'M' else 0) #Just gender
        df_enc1 = pd.concat([df_demog[0],demogra_enc], axis=1) #Adding back filenames to the DF
        df_enc1[0] = df_enc1[0].apply(lambda x: x.replace('WAV', 'wav')) #Dealing with incorrect capitalization

        #Setting the filenames as the index
        demogra_enc = df_enc1.set_index(0, drop=True)

        demogra_enc_train = demogra_enc.iloc[0:8460,:]
        demogra_enc_dev = demogra_enc.iloc[8460:9360,:]
        demogra_enc_test = demogra_enc.iloc[9360:12360,:]
        print('Train Encoded Shape:', demogra_enc_train.shape, 'Dev Encoded Shape:', demogra_enc_dev.shape, 'Test Encoded Shape:', demogra_enc_test.shape,)

        return demogra_enc_train, demogra_enc_dev, demogra_enc_test


    def process_audio_and_save_npy_DemograChallenge(self, data_file, demogra_df, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file, header=None, delimiter='\t')
        
        print('shape of final deomg_enc_df', demogra_df.shape)

        print('Number of audio files ', len(df))
        if shuffle:
            df = df.sample(frac=1)

        # Converting 'A' to 1 and 'N' to 0 - according to Challenge's binary decision.
        # Refer Challenge's Readme for further information
        df[1] = df[1].apply(lambda x: 1 if x == 'A' else 0)
        # Irregular use of extensions in data, so handling it here
        df[0] = df[0].apply(lambda x: x.replace('WAV', 'wav'))

        data, labels, demographics = preprocess_data_with_demogra(self.base_path, df[0].values, df[1].values, demogra_df,
                                       self.normalise,
                                       self.sample_size_in_seconds, self.sampling_rate, self.overlap, self.method)
        print('Number of audio files after processing ', len(data))
        save_npy(data, self.data_save_path  + filename_to_save + '_data.npy')
        save_npy(labels, self.data_save_path + filename_to_save + '_labels.npy')
        save_npy(demographics, self.data_save_path + filename_to_save + '_demographics.npy')

    def process_audio_and_save_csv(self, data_file, filename_to_save, shuffle=True):
        df = pd.read_csv(data_file)
        if shuffle:
            df = df.sample(frac=1)
        data, labels = preprocess_data_images(self.base_path, self.image_save_path, df['WAV_PATH'].values,
                                              df['label'].values,
                                              self.normalise,
                                              self.sample_size_in_seconds, self.sampling_rate, self.overlap)
        concat_data = np.concatenate((np.array([data]).T, np.array([labels]).T), axis=1)
        save_csv(concat_data, columns=["spectrogram_path", "labels"], filename=
        self.data_save_path + '/' + filename_to_save + '_data_melfilter_specs.csv')

    def run(self):
        demogra_enc_train, demogra_enc_dev, demogra_enc_test = self.one_hot_Encoder()
        print('Started processing train data . . .')
        self.process_audio_and_save_npy_DemograChallenge(self.train_data_file, demogra_enc_train, filename_to_save='train_challenge_with_d1')
        print('Started processing dev data . . .')
        self.process_audio_and_save_npy_DemograChallenge(self.dev_data_file, demogra_enc_dev, filename_to_save='dev_challenge_with_d1')
        print('Started processing test data . . .')
        self.process_audio_and_save_npy_DemograChallenge(self.test_data_file, demogra_enc_test, filename_to_save='test_challenge_with_d1')


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)

    processor = DataProcessor(configs)
    print(configs)
    processor.run()
