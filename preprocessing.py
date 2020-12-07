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


def preprocessing2(train_features, train_labels, valid_features, valid_labels, test_features):
    lab = np.array([
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
        'street_music'
    ])
    # Label encoder
    le = LabelEncoder().fit(lab)
    lab = le.transform(lab).reshape(-1, 1)
    train_labels = le.transform(train_labels).reshape(-1, 1)
    valid_labels = le.transform(valid_labels).reshape(-1, 1)
    # onehot encoder
    enc = OneHotEncoder().fit(lab)
    train_labels = enc.transform(train_labels).toarray()
    valid_labels = enc.transform(valid_labels).toarray()

    # # 正規化
    stan = StandardScaler().fit(train_features)
    train_features = stan.transform(train_features)
    valid_features = stan.transform(valid_features)
    test_features = stan.transform(test_features)

    return train_features, train_labels, valid_features, valid_labels, test_features