# Load various imports
import pandas as pd
import os
import librosa
import numpy as np


def mfcc(file_name):

    try:
        audio, sample_rate = librosa.load(
            file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        # print(mfccsscaled.shape)
        
        print('mfccs.shape = ', mfccs.shape)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None

    return mfccs


def feature_extraction():

    # Set the path to the full UrbanSound dataset
    fulldatasetpath = 'D:/DATASET/UrbanSound8K/audio'

    metadata = pd.read_csv('metadata/UrbanSound8K.csv')

    features = []

    # Iterate through each sound file and extract the features
    i = 0
    for index, row in metadata.iterrows():
        # 讀取前400筆資料
        i += 1
        if i > 2:
            break
        file_name = os.path.join(os.path.abspath(
            fulldatasetpath), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))

        class_label = row["class"]
        data = mfcc(file_name)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    return featuresdf
