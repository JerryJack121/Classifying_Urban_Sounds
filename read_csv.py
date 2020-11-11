import pandas as pd
import numpy as np
import librosa
import pandas as pd

def read_csv():
    labels = pd.read_csv('csv/labels.csv')
    mfcc = pd.read_csv('csv/mfcc.csv', header = None)
    chromagram = pd.read_csv('csv/chromagram.csv', header = None)
    tonnetz = pd.read_csv('csv/tonnetz.csv', header = None)
    mel = pd.read_csv('csv/mel.csv', header = None)

    mfcc =  np.array(mfcc)
    mel =  np.array(mel)
    chromagram =  np.array(chromagram)
    tonnetz =  np.array(tonnetz)


    #矩陣水平合併
    feature = np.hstack((mfcc,mel,chromagram,tonnetz))
    labels =  np.array(labels)

    return feature,labels