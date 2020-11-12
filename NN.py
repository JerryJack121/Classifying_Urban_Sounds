from keras.models import Sequential, load_model
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
                        batch_size=64, epochs=10,
                        validation_data=(x_test, y_test))

    model.save('model_1000.h5')
    return history


def plot_history(history):
    import matplotlib.pyplot as plt

    history_dict = history.history
    lose_values = history_dict['loss']
    val_lose_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(lose_values)+1)
    # 繪製訓練與驗證的損失分數
    plt.figure()
    plt.subplot(211)
    plt.plot(epochs, lose_values, 'bo', label='Training loss')
    plt.plot(epochs, val_lose_values, 'b', label='Validation loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 繪製訓練與驗證的準確度
    plt.subplot(212)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def model_test(x_test, y_test):
    # 從 HDF5 檔案中載入模型
    model = load_model('model_1000.h5')
    # 驗證模型
    score = model.evaluate(x_test, y_test, verbose=0)

    # 輸出結果
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])