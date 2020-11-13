from feature2csv import feature2csv
from read_csv import read_csv
from split_data import split_data
from NN import training, model_test


# feature2csv()
#讀取儲存在csv中的data
feature, labels = read_csv()
#預處理
x_train, y_train, x_val, y_val, y_test, y_test = split_data(feature, labels)
#訓練模型
training(x_train, x_val, y_train, y_val)
