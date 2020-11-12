from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def NN_model():
    model = Sequential()

    model.add(Dense(160, input_shape=(160,), activation='relu'))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')
    return model


def training(x_train, x_test, y_train, y_test):
    model = NN_model()
    history = model.fit(x_train, y_train,
                        batch_size=64, epochs=100,
                        validation_data=(x_test, y_test))

    model.save('model_1000.h5')