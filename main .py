from read_csv import read_csv
from split_data import split_data
from train import training


feature,labels = read_csv()
x_train,x_test,y_train,y_test = split_data(feature,labels)
training(x_train,x_test,y_train,y_test)