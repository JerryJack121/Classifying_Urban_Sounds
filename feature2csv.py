import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa

data = pd.read_csv('D:/DATASET/UrbanSound8K/train.csv')
print(data.head())  # To see the dataset

mfc = []
lab = []
ton = []
me = []
chro = []
# for i in tqdm(range(len(data))):

for i in tqdm(range(10)):
    f_name = 'D:/DATASET/UrbanSound8K/Train/'+str(data.ID[i+1])+'.wav'
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

# mfcc = pd.DataFrame(mfc)
# mfcc.to_csv('./csv/mfcc.csv', 'w', index=False, header=False)
# chromagram = pd.DataFrame(chro)
# chromagram.to_csv('./csv/chromagram.csv', 'w', index=False, header=False)
# mel = pd.DataFrame(me)
# mel.to_csv('./csv/mel.csv', 'w', index=False, header=False)
# tonnetz = pd.DataFrame(ton)
# tonnetz.to_csv('./csv/tonnetz.csv', 'w', index=False, header=False)
# lab_dict = {'classID': lab}
# la = pd.DataFrame(lab_dict)
# la.to_csv('./csv/labels.csv', 'w', index=False, header=False)


feature_dict = {'classID': lab, 'mel': me,  'mfcc': mfc,
                'chromagram': chro, 'tonnetz': ton}
featuredf = pd.DataFrame(feature_dict)
featuredf.to_csv('./csv/feature.csv')
print('CSV寫入完成')