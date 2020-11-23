from read_csv import read_csv
from preprocessing import preprocessing
from train import train

#讀取儲存在csv中的data
feature, labels = read_csv()
#預處理
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(feature, labels)
#訓練模型
train(x_train, x_val, y_train, y_val, batch_size=64, epochs=50)
