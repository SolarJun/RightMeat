import matplotlib.pyplot as plt

from skimage import io


class RGB_Histogram():
    def __init__(self):
        # Fresh
        self.image_Fresh = io.imread(
            './archive/Fresh/test_20171016_104321D.jpg')

        # Spoiled
        self.image_Spoiled = io.imread(
            './archive/Spoiled/test_20171017_190121D.jpg')

    def Draw(self):
        # Fresh
        image_Fresh = self.image_Fresh

        # ravel() : numpy 다차원 배열을 1차원으로 변환
        plt.hist(image_Fresh.ravel(), bins=256, color='gray')
        plt.hist(image_Fresh[:, :, 0].ravel(),
                 color='red', bins=256, alpha=0.5)
        plt.hist(image_Fresh[:, :, 1].ravel(),
                 color='Green', bins=256, alpha=0.5)
        plt.hist(image_Fresh[:, :, 2].ravel(),
                 color='Blue', bins=256, alpha=0.5)
        plt.title('Fresh Meat RGB Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend(['Total', 'Red', 'Green', 'Blue'])
        plt.show()

        # Spoiled
        image_Spoiled = self.image_Spoiled

        plt.hist(image_Spoiled.ravel(), bins=256, color='gray')
        plt.hist(image_Spoiled[:, :, 0].ravel(),
                 color='red', bins=256, alpha=0.5)
        plt.hist(image_Spoiled[:, :, 1].ravel(),
                 color='Green', bins=256, alpha=0.5)
        plt.hist(image_Spoiled[:, :, 2].ravel(),
                 color='Blue', bins=256, alpha=0.5)
        plt.title('Spoiled Meat RGB Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend(['Total', 'Red', 'Green', 'Blue'])
        plt.show()


RGB_Histogram().Draw()
