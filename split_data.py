from read_csv import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np


def split_data(feature, labels):
    lab = ['air_conditioner', ' car_horn', 'children_playing', 'dog_bark',
           'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    le = LabelEncoder()
    le = le.fit(lab)
    # Label encoder
    labels = np.array(labels)
    lab = le.transform(lab)
    label = le.transform(labels)
    # onehot encoder
    lab = lab.reshape(-1, 1)
    label = label.reshape(-1, 1)
    enc = OneHotEncoder().fit(lab)
    label = enc.transform(label).toarray()
    print(label)

    # 正規化
    stan = StandardScaler()
    stan = stan.fit(feature)
    feature = stan.transform(feature)

    x_train = feature[:5]
    x_test = feature[5:]
    y_train = label[:5]
    y_test = label[5:]

    return x_train, x_test, y_train, y_test
