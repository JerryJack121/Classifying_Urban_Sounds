from Net.NN import NN_model
import matplotlib.pyplot as plt


def plot_history(history):
    history_dict = history.history
    lose_values = history_dict['loss']
    val_lose_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(lose_values) + 1)
    # 繪製訓練與驗證的損失分數
    plt.figure()
    plt.subplot(211)
    plt.plot(epochs, lose_values, 'r', label='Training loss')
    plt.plot(epochs, val_lose_values, 'b', label='Validation loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 繪製訓練與驗證的準確度
    plt.subplot(212)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train(x_train, x_test, y_train, y_test, batch_size=64, epochs=10):
    model = NN_model(feature_size = x_train.shape[-1])
    history = model.fit(x_train,
                        y_train,
                        batch_size,
                        epochs,
                        validation_data=(x_test, y_test))

    save_name = './model/model_1207.h5'
    model.save(save_name)
    print(save_name, '存檔')
    plot_history(history)