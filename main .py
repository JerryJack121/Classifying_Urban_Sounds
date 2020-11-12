from read_csv import read_csv
from split_data import split_data
from NN import training, plot_history


feature, labels = read_csv()
x_train, y_train, x_val, y_val, y_test, y_test = split_data(feature, labels)
history = training(x_train, x_val, y_train, y_val)
plot_history(history)
