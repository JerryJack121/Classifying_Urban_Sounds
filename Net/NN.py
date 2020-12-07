from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def NN_model(feature_size):
    model = Sequential()

    model.add(Dense(feature_size, input_shape=(feature_size, ), activation='relu'))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')
    return model
