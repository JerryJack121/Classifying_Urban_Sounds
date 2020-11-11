from read_csv import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np


feature, labels = read_csv()
le = LabelEncoder()
le = le.fit(['air_conditioner', ' car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])

labels = np.array(labels)
label = le.transform(labels)

x_train = feature[:5]
x_test = feature[5:]
y_train = label[5:]
y_test