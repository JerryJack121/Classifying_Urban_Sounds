import pandas as pd
import numpy as np
import librosa
import pandas as pd

def read_csv():
    labels = pd.read_csv('csv/train/labels.csv', header = None)
    mfcc = pd.read_csv('csv/mfcc.csv', header = None)
    # chromagram = pd.read_csv('csv/chromagram.csv', header = None)
    # tonnetz = pd.read_csv('csv/tonnetz.csv', header = None)
    # mel = pd.read_csv('csv/mel.csv', header = None)

    mfcc =  np.array(mfcc)
    mel =  np.array(mel)
    chromagram =  np.array(chromagram)
    tonnetz =  np.array(tonnetz)


    #矩陣水平合併
    feature = np.hstack((mfcc,mel,chromagram))
    labels =  np.array(labels)
    labels = labels.ravel()

    return feature,labels

def read_csv2():
    train_labels = pd.read_csv('csv/train/labels.csv', header = None)
    train_features = pd.read_csv('csv/train/mfcc.csv', header = None)
    test_features = pd.read_csv('csv/test/mfcc.csv', header = None)

    return train_features, train_labels, test_features