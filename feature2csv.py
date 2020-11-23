# 特徵提取並儲存於csv中
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa

def feature2csv():
    data = pd.read_csv('D:/DATASET/UrbanSound8K/train.csv')
    print(data.head())  # To see the dataset

    mfc = []
    lab = []
    ton = []
    me = []
    chro = []
    ID = []

    # for i in tqdm(range(len(data))):
    for i in tqdm(range(len(data))):
        f_name = 'D:/DATASET/UrbanSound8K/Train/'+str(data.ID[i])+'.wav'
        ID.append(data.ID[i])
        X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
        # mfccs
        mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T, axis=0)
        mfc.append(mf)
        l = data.Class[i]
        lab.append(l)
        # tonnetz
        try:
            t = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=s_rate).T, axis=0)
            ton.append(t)
        except:
            print(f_name)
        # mel-scaled spectrogram
        m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T, axis=0)
        me.append(m)
        # chromagram
        s = np.abs(librosa.stft(X))
        c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T, axis=0)
        chro.append(c)

    mfcc = pd.DataFrame(mfc)
    mfcc.to_csv('./csv/mfcc.csv', index=False, header=False)
    chromagram = pd.DataFrame(chro)
    chromagram.to_csv('./csv/chromagram.csv', index=False, header=False)
    mel = pd.DataFrame(me)
    mel.to_csv('./csv/mel.csv', index=False, header=False)
    tonnetz = pd.DataFrame(ton)
    tonnetz.to_csv('./csv/tonnetz.csv', index=False, header=False)
    lab_dict = {'classID': lab}
    la = pd.DataFrame(lab_dict)
    la.to_csv('./csv/labels.csv', index=False, header=False)

    print('CSV寫入完成')

if __name__ == "__main__":
    feature2csv()
