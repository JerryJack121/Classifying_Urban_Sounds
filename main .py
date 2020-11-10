import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa

data=pd.read_csv('D:/DATASET/UrbanSound8K/train.csv')
print(data.head())     #To see the dataset

mfc=[]
lab=[]
# for i in tqdm(range(len(data))):
for i in tqdm(range(100)):
    f_name = 'D:/DATASET/UrbanSound8K/Train/'+str(data.ID[i])+'.wav'
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
    mfc.append(mf)
    l=data.Class[i]
    lab.append(l)