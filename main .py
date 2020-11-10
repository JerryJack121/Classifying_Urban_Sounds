from helpers.load_data import load_data
from helpers.feature_extraction import feature_extraction
from helpers.convert_data import convert_data
from helpers.model import cnn

# audiodf = load_data()

featuresdf = feature_extraction()
feature = featuresdf.feature
# print(feature[0].shape)
#x_train, x_test, y_train, y_test = convert_data(featuresdf)
# cnn(x_train, x_test, y_train, y_test)