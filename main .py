from read_csv import read_csv
from split_data import split_data
from train import training, plot_history


feature,labels = read_csv()
x_train,x_test,y_train,y_test = split_data(feature,labels)
history = training(x_train,x_test,y_train,y_test)
plot_history(history)