import numpy as np
import matplotlib.pyplot as plt
import Data_Preprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class MLP():
    def __init__(self):
        data_preprocessing = Data_Preprocessing.Data_Preprocessing()
        self.trainx, self.testx, self.trainy, self.testy = data_preprocessing.Encoding_Split()

    def Data(self):
        trainx, testx, trainy, testy = self.trainx, self.testx, self.trainy, self.testy

        print(np.shape(trainx))  # (1516, 960, 480, 3)
        print(np.shape(testx))  # (380, 960, 480, 3)
        print(np.shape(trainy))  # (1516, )
        print(np.shape(testy))  # (380, )

        trainx = np.array(trainx)  # list -> numpy array
        testx = np.array(testx)  # list -> numpy array

        trainx = trainx.reshape((trainx.shape[0], -1))  # (장 수, 픽셀*픽셀*3)
        testx = testx.reshape((testx.shape[0], -1))  # (장 수, 픽셀*픽셀*3)

        # to_categorical : one-hot encoding 해주는 함수
        trainy = to_categorical(trainy, 2)
        testy = to_categorical(testy, 2)

        print(trainy)
        print(trainy.shape)

        return trainx, testx, trainy, testy

    def Model_Fit(self):
        trainx, testx, trainy, testy = self.Data()

        # 신경망 구조 설계
        print(trainx.shape[-1])
        n_input = trainx.shape[-1]
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

        print(model.input_shape)
        print(model.output_shape)

        # 신경망 학습
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            learning_rate=0.001), metrics=['accuracy'])
        hist = model.fit(trainx, trainy[:trainy.shape[0]], batch_size=64, epochs=30,
                         validation_data=(testx, testy), verbose=2)

        return model, hist

    def Model_Prediction(self):
        trainx, testx, trainy, testy = self.Data()
        model, hist = self.Model_Fit()

        # 학습된 신경망으로 예측
        result = model.evaluate(testx, testy)
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


MLP().Model_Prediction()
