from read_csv import read_csv, read_csv2
from preprocessing import preprocessing
from train import train

#讀取儲存在csv中的data
# feature, labels = read_csv()
train_features, train_labels, test_features = read_csv2()

#切割資料集
valid_features = train_features[3500:,:]
valid_labels = train_labels[3500:,:]
train_features = train_features[:3500,:]
train_labels = train_labels[:3500,:]

#預處理
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(feature, labels)
#訓練模型
train(x_train, x_val, y_train, y_val, batch_size=64, epochs=50)
