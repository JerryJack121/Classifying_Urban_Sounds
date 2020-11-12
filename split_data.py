from read_csv import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np


def split_data(feature, labels):
    lab = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
           'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    lab =  np.array(lab)
    lab = lab.ravel()
    # Label encoder
    le = LabelEncoder().fit(lab)
    le.transform(labels)
    # onehot encoder
    lab = lab.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder().fit(lab)
    labels = enc.transform(labels).toarray()

    # # 正規化
    stan = StandardScaler()
    stan = stan.fit(feature)
    feature = stan.transform(feature)

    x_train = feature[:4000]
    y_train = labels[:4000]
    x_test = feature[4000:5000]
    y_test = labels[4000:5000]

    return x_train, x_test, y_train, y_test
