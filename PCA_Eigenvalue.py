'''
PCA_Eigenvalue ploting
- Image Size : 10x10, 5x5
'''

import numpy as np
import matplotlib.pyplot as plt
import Data_Preprocessing

from sklearn.decomposition import PCA


class PCA():
    def __init__(self):
        data_preprocessing = Data_Preprocessing.Data_Preprocessing()
        self.trainx, self.testx, self.trainy, self.testy = data_preprocessing.Encoding_Split()

    def EigenValue(self):
        trainx, testx, trainy, testy = self.trainx, self.testx, self.trainy, self.testy

        print(trainx.shape)  # (1516, 960, 480, 3)
        print(testx.shape)  # (380, 960, 480, 3)
        print(trainy.shape)  # (1516, 2)
        print(testy.shape)  # (380, 2)

        trainx = trainx.reshape((trainx.shape[0], -1))  # (장수, 픽셀*픽셀*3)
        print(trainx.shape)  # (장수, pixel * pixel * 3)

        trainx_m = trainx - np.mean(trainx, axis=0)
        print(trainx_m)
        print(trainx_m.shape)

        cov_T = np.cov(trainx_m.T)
        print(cov_T.shape)

        eig_val, eig_vec = np.linalg.eig(cov_T)
        plt.plot(eig_val)
        plt.show()


PCA().EigenValue()
