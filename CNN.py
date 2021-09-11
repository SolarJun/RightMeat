import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Data_Preprocessing

from tensorflow.keras import layers, models


class CNN():
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

        return trainx, testx, trainy, testy

    def Model_Fit(self):
        trainx, testx, trainy, testy = self.Data()
        print(trainx[-1].shape)

        # MLP hidden_node=50,25 :: CNN (32, 32, 32), (16, 32, 16)
        model = models.Sequential()  # layer가 순차적으로 쌓여가는 것을 의미
        model.add(layers.Conv2D(16, (3, 3), activation='relu',
                                input_shape=trainx[-1].shape))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2))

        print(model.input_shape)
        print(model.output_shape)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(trainx, trainy[:trainy.shape[0]], batch_size=64, epochs=30,
                            validation_data=(testx, testy))

        return model, history

    def Model_Prediction(self):
        trainx, testx, trainy, testy = self.Data()
        model, history = self.Model_Fit()

        # loss, accuracy
        result = model.evaluate(testx, testy)
        for i in range(len(model.metrics_names)):
            print(model.metrics_names[i], ":", result[i]*100)

        print(model.summary())

        # ploting
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model accuracy/loss')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train_Acc', 'Val_Acc', 'Train_Loss',
                    'Val_Loss'], loc='upper left')
        plt.grid()
        plt.show()

    def KernelMap(self):
        model, history = self.Model_Fit()

        # KernelMap
        for layer in model.layers:  # 컨볼루션층의 커널을 시각화
            if'conv' in layer.name:
                kernel, biases = layer.get_weights()
                print(layer.name, kernel.shape)  # 커널의 텐서 모양을 출력

        kernel, biases = model.layers[0].get_weights()  # 층 0의 커널 정보를 저장
        minv, maxv = kernel.min(), kernel.max()  # 맨 앞에 있는 컨볼루션층의 커널 정보를 추출
        kernel = (kernel-minv)/(maxv-minv)
        n_kernel = 16

        plt.suptitle("Kernels of conv2d_4")
        for i in range(n_kernel):  # i번째 커널
            f = kernel[:, :, :, i]
            for j in range(3):  # j번째 채널
                plt.subplot(3, n_kernel, j*n_kernel+i+1)
                plt.imshow(f[:, :, j], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.title(str(i)+'_'+str(j))
        plt.show()

    def FeatureMap(self):
        trainx, testx, trainy, testy = self.Data()
        model, history = self.Model_Fit()

        # FeatureMap
        for layer in model.layers:  # 특징 맵의 탠서 모양을 출력
            if 'conv' in layer.name:
                print(layer.name, layer.output.shape)

        partial_model = models.Model(inputs=model.inputs,
                                     outputs=model.layers[0].output)  # 층 0만 떼어냄
        partial_model.summary()

        feature_map = partial_model.predict(testx)  # 부분 모델로 테스트 집합을 예측

        fm = feature_map[1]  # 1번 영상의 특징 맵을 시각화
        plt.imshow(testx[1])  # 1번 영상을 출력
        plt.show()

        plt.suptitle("Feature maps of conv2d_4")
        for i in range(16):  # i번째 특징 맵
            plt.subplot(2, 8, i+1)
            plt.imshow(fm[:, :, i], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("map"+str(i))
        plt.show()


CNN().Model_Prediction()
CNN().KernelMap()
CNN().FeatureMap()
