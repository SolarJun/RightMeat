import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Data_Image_Crop_Preprocessing():
    def __init__(self):
        self.directory = './archive'

        self.Name = []
        for file in os.listdir(self.directory):  # 각 이미지 파일만
            if file[-4:] != '.txt':
                self.Name += [file]
        # print(Name)  # Fresh, Spoiled 파일
        # print(len(Name))  # 2개 파일

        self.dataset_crop = []
        self.image_target = []
        for name in self.Name:  # Fresh, Spoiled
            path = os.path.join(self.directory, name)

            for image_name in os.listdir(path):
                if image_name[-4:] == '.jpg':
                    image = load_img(os.path.join(
                        path, image_name), grayscale=False, color_mode='rgb', target_size=(720, 1280))

                    # 이미지 픽셀 크기 확인
                    # print(image.size) # (1280, 720) -> (960, 480) 변환 : 원래 크기는 RAM 메모리 부족으로 안 됨

                    # MLP, CNN 실험할 해상도 : 각각 16x9, 32x18
                    area = (image.width/4, image.height/4,
                            image.width/4 + 16, image.height/4 + 9)
                    image_crop = image.crop(area)

                    # 픽셀별 이미지 해상도 확인
                    # plt.imshow(image)
                    # ax = plt.gca()
                    # crop_rect = patches.Rectangle((image.width/4, image.height/4),
                    #                               16, 9,
                    #                               linewidth=2,
                    #                               edgecolor='yellow',
                    #                               fill=False)
                    # ax.add_patch(crop_rect)
                    # plt.show()

                    # plt.imshow(image_crop)
                    # plt.show()

                    # break

                    # print(np.amin(image))  # 이미지 픽셀 최소값 : 0.0
                    # print(np.amax(image))  # 이미지 픽셀 최대값 : 255.0
                    # print(np.amin(image_crop))  # 이미지 픽셀 최소값 : 0.0
                    # print(np.amax(image_crop))  # 이미지 픽셀 최대값 : 238.0
                    # break

                    image_crop = img_to_array(image_crop)
                    image_crop = image_crop/255.0

                    self.dataset_crop.append(image_crop)
                    self.image_target.append(name)

                    # break

    def Encoding_Split(self):
        # labels
        labels = LabelEncoder()  # 0과 n_classes-1 사이의 값으로 target을 인코딩
        labels = labels.fit(self.image_target)
        labels = labels.transform(self.image_target)

        crop_trainx, crop_testx, crop_trainy, crop_testy = train_test_split(
            self.dataset_crop, labels, test_size=0.4, random_state=42)
        # random_state : set 중 42개씩 섞으며, 매번 데이터셋이 변경되는 것을 방지

        return crop_trainx, crop_testx, crop_trainy, crop_testy
