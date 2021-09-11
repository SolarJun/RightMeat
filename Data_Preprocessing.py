import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data_Preprocessing():
    def __init__(self):
        self.directory = './archive'

        self.Name = []
        for file in os.listdir(self.directory):  # 각 이미지 파일만
            if file[-4:] != '.txt':
                self.Name += [file]
        # print(Name)  # Fresh, Spoiled 파일
        # print(len(Name))  # 2개 파일

        self.dataset = []
        self.image_target = []
        for name in self.Name:  # Fresh, Spoiled
            path = os.path.join(self.directory, name)

            for image_name in os.listdir(path):
                if image_name[-4:] == '.jpg':
                    # MLP, CNN 최고 성능 : 각각 9X16, 18X32
                    image = load_img(os.path.join(
                        path, image_name), grayscale=False, color_mode='rgb', target_size=(9, 16))

                    # 이미지 픽셀 크기 확인
                    # print(image.size) # (1280, 720) -> (960, 480) 변환 : 원래 크기는 RAM 메모리 부족으로 안 됨

                    # 픽셀별 이미지 해상도 확인
                    # plt.imshow(image)
                    # plt.show()

                    # break

                    image = img_to_array(image)  # image를 numpy 배열로 변환
                    # print(np.amin(image)) # 이미지 픽셀 최소값 : 0.0
                    # print(np.amax(image))  # 이미지 픽셀 최대값 : 255.0

                    image = image/255.0  # 픽셀값을 0~1의 값으로 정규화

                    self.dataset.append(image)
                    self.image_target.append(name)

    def Encoding_Split(self):
        labels = LabelEncoder()  # 0과 n_classes-1 사이의 값으로 target을 인코딩
        labels = labels.fit(self.image_target)
        labels = labels.transform(self.image_target)

        trainx, testx, trainy, testy = train_test_split(
            self.dataset, labels, test_size=0.4, random_state=42)
        # random_state : set 중 42개씩 섞으며, 매번 데이터셋이 변경되는 것을 방지

        return trainx, testx, trainy, testy
