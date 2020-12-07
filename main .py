from read_csv import read_csv, read_csv2
from preprocessing import preprocessing, preprocessing2
from train import train

#讀取儲存在csv中的data
# feature, labels = read_csv()
train_features, train_labels, test_features = read_csv2()

#切割資料集
valid_features = train_features[3500:, :]
valid_labels = train_labels[3500:, :]
train_features = train_features[:3500, :]
train_labels = train_labels[:3500, :]

#預處理
train_features, train_labels, valid_features, valid_labels, test_features = preprocessing2(
    train_features, train_labels, valid_features, valid_labels, test_features)


#訓練模型
train(train_features, valid_features, train_labels, valid_labels, batch_size=64, epochs=100)
