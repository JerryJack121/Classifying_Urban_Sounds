from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
import numpy as np

def convert_data(featuresdf):
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 
    #print(yy)
    
    # split the dataset 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    return(x_train, x_test, y_train, y_test)