from read_csv import read_csv
from preprocessing import preprocessing
from keras.models import load_model
import numpy as np


lab = np.array([
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
    'street_music'
])

#讀取儲存在csv中的data
feature, labels = read_csv()
#預處理
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(feature, labels)
# 從 HDF5 檔案中載入模型
model = load_model('./model/model_1123.h5')
# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict = model.predict_classes(x_test)  #To predict labels

# 第0筆預測資料對應的label
print(lab[predict[0:5]])