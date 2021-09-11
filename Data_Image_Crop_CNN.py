import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Data_Image_Crop_Preprocessing

from tensorflow.keras import models, layers


class Data_Image_Crop_CNN():
    def __init__(self):
        data_image_crop_preprocessing = Data_Image_Crop_Preprocessing.Data_Image_Crop_Preprocessing()
        self.crop_trainx, self.crop_testx, self.crop_trainy, self.crop_testy = data_image_crop_preprocessing.Encoding_Split()

    def Data(self):
        crop_trainx, crop_testx, crop_trainy, crop_testy = self.crop_trainx, self.crop_testx, self.crop_trainy, self.crop_testy

        print(np.shape(crop_trainx))  # (1137, 18, 32, 3)
        print(np.shape(crop_testx))  # (759, 18, 32, 3)
        print(np.shape(crop_trainy))  # (1137, )
        print(np.shape(crop_testy))  # (759, )

        crop_trainx = np.array(crop_trainx)  # list -> numpy array
        crop_testx = np.array(crop_testx)  # list -> numpy array

        return crop_trainx, crop_testx, crop_trainy, crop_testy

    def Model_Fit(self):
        crop_trainx, crop_testx, crop_trainy, crop_testy = self.Data()

        # MLP hidden_node=50,25 :: CNN (32, 32, 32), (16, 32, 16)
        model = models.Sequential()  # layer가 순차적으로 쌓여가는 것을 의미
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=crop_trainx[-1].shape))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(crop_trainx, crop_trainy[:crop_trainy.shape[0]], batch_size=64, epochs=30,
                            validation_data=(crop_testx, crop_testy))

        return model, history

    def Model_Prediction(self):
        crop_testx, crop_testx, crop_trainy, crop_testy = self.Data()
        model, history = self.Model_Fit()

        # loss, accuracy
        result = model.evaluate(crop_testx, crop_testy)
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


Data_Image_Crop_CNN().Model_Prediction()
