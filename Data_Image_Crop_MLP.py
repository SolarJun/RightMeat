import numpy as np
import os
import matplotlib.pyplot as plt
import Data_Image_Crop_Preprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class Data_Image_Crop_MLP():
    def __init__(self):
        data_image_crop_preprocessing = Data_Image_Crop_Preprocessing.Data_Image_Crop_Preprocessing()
        self.crop_trainx, self.crop_testx, self.crop_trainy, self.crop_testy = data_image_crop_preprocessing.Encoding_Split()

    def Data(self):
        crop_trainx, crop_testx, crop_trainy, crop_testy = self.crop_trainx, self.crop_testx, self.crop_trainy, self.crop_testy

        print(np.shape(crop_trainx))  # (1137, 9, 16, 3)
        print(np.shape(crop_testx))  # (759, 9, 16, 3)
        print(np.shape(crop_trainy))  # (1137, )
        print(np.shape(crop_testy))  # (759, )

        crop_trainx = np.array(crop_trainx)  # list -> numpy array
        crop_testx = np.array(crop_testx)  # list -> numpy array

        crop_trainx = crop_trainx.reshape(
            (crop_trainx.shape[0], -1))  # (장 수, 픽셀*픽셀*3)
        crop_testx = crop_testx.reshape(
            (crop_testx.shape[0], -1))  # (장 수, 픽셀*픽셀*3)

        # to_categorical : one-hot encoding 해주는 함수
        crop_trainy = to_categorical(crop_trainy, 2)
        crop_testy = to_categorical(crop_testy, 2)

        print(crop_trainy)
        print(crop_trainy.shape)

        return crop_trainx, crop_testx, crop_trainy, crop_testy

    def Model_Fit(self):
        crop_trainx, crop_testx, crop_trainy, crop_testy = self.Data()

        # 신경망 구조 설계
        print(crop_trainx.shape[-1])
        n_input = crop_trainx.shape[-1]
        n_hidden1 = 25
        # n_hidden2 = 200
        # n_hidden3 = 256
        # n_hidden4 = 128
        n_output = 2

        # 신경망 구조 설계
        model = Sequential()
        model.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,),
                        kernel_initializer='random_uniform', bias_initializer='zeros'))
        # mlp.add(Dense(units=n_hidden2, activation='relu',
        #               kernel_initializer='random_uniform', bias_initializer='zeros'))
        # mlp.add(Dense(units=n_hidden3, activation='tanh',
        #               kernel_initializer='random_uniform', bias_initializer='zeros'))
        # mlp.add(Dense(units=n_hidden4, activation='tanh',
        #               kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(units=n_output, activation='softmax',
                        kernel_initializer='random_uniform', bias_initializer='zeros'))

        # 신경망 학습
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            learning_rate=0.001), metrics=['accuracy'])
        hist = model.fit(crop_trainx, crop_trainy[:crop_trainy.shape[0]], batch_size=64, epochs=30,
                         validation_data=(crop_testx, crop_testy), verbose=2)

        return model, hist

    def Model_Prediction(self):
        crop_trainx, crop_testx, crop_trainy, crop_testy = self.Data()
        model, hist = self.Model_Fit()

        # 학습된 신경망으로 예측
        result = model.evaluate(crop_testx, crop_testy)
        for i in range(len(model.metrics_names)):
            print(model.metrics_names[i], ":", result[i]*100)

        # 정확률 곡선
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model accuracy/loss')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train_Acc', 'Val_Acc', 'Train_Loss',
                    'Val_Loss'], loc='upper left')
        plt.grid()
        plt.show()

        print(model.summary())

        '''
        파리미터 수 계산
        - 입력 : n 
        - 히든노드 : h 
        - 출력노드 : m
        - 파라미터(즉 가중치 갯수) : (n+1)×h + (h+1)×m
        '''


Data_Image_Crop_MLP().Model_Prediction()
