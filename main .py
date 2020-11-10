from helpers.load_data import load_data
from helpers.feature_extraction import feature_extraction
from helpers.convert_data import convert_data


# audiodf = load_data()

featuresdf = feature_extraction()

x_train, x_test, y_train, y_test = convert_data(featuresdf)
