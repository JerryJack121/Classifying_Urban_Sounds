from read_csv import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np


def preprocessing(feature, labels):
    lab = np.array([
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
        'street_music'
    ])
    # Label encoder
    le = LabelEncoder().fit(lab)
    lab = le.transform(lab).reshape(-1, 1)
    labels = le.transform(labels).reshape(-1, 1)
    # onehot encoder
    enc = OneHotEncoder().fit(lab)
    labels = enc.transform(labels).toarray()

    # # 正規化
    stan = StandardScaler()
    feature = stan.fit_transform(feature)

    x_train = feature[:4000]
    y_train = labels[:4000]
    x_val = feature[4000:5000]
    y_val = labels[4000:5000]
    x_test = feature[5000:]
    y_test = labels[5000:]

    return x_train, y_train, x_val, y_val, x_test, y_test
