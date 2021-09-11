import matplotlib.pyplot as plt
import numpy as np

Image_size = ['16X9', '32X18', '48X27', '64X36', '80X45', '96X54']

MLP_hidden_node_50_Acc = [99.82425570487976, 99.82425570487976, 98.41827750205994,
                          99.4727611541748, 98.94551634788513, 99.82425570487976]
MLP_hidden_node_50_Loss = [2.542359009385109, 1.7222622409462929, 2.6155073195695877,
                           1.9984470680356026, 2.6662489399313927, 1.4496962539851665]

MLP_hidden_node_25_Acc = [99.82425570487976, 99.4727611541748, 98.41827750205994,
                          99.82425570487976, 99.4727611541748, 99.64850544929504]
MLP_hidden_node_25_Loss = [4.106833040714264, 3.13604436814785, 3.3555205911397934,
                           1.5284352004528046, 1.5727261081337929, 2.089094929397106]

CNN_hidden_node_50_Acc = [98.81423115730286, 99.73649382591248, 99.60474371910095,
                          98.81423115730286, 98.81423115730286, 98.81423115730286]
CNN_hidden_node_50_Loss = [2.7620408684015274, 1.2052186764776707, 1.1952399276196957,
                           2.7620408684015274, 2.7620408684015274, 2.7620408684015274]

CNN_hidden_node_25_Acc = [99.47299361228943, 99.73649382591248, 99.2094874382019,
                          98.81423115730286, 99.47299361228943, 99.73649382591248]
CNN_hidden_node_25_Loss = [1.1986928060650826, 1.180399302393198, 1.6206897795200348,
                           2.7620408684015274, 1.1110922321677208, 1.2387641705572605]

MLP_hidden_node_25_Acc_SIZE_16_9 = 99.82425570487976
MLP_Crop_hidden_node_25_Acc_SIZE_16_9 = 97.62845635414124
CNN_hidden_node_25_Acc_SIZE_32_18 = 99.73649382591248
CNN_Crop_hidden_node_25_Acc_SIZE_32_18 = 98.68247509002686
MLP_CNN_Crop_Compared = [CNN_hidden_node_25_Acc_SIZE_32_18,
                         CNN_Crop_hidden_node_25_Acc_SIZE_32_18,
                         CNN_hidden_node_25_Acc_SIZE_32_18,
                         CNN_Crop_hidden_node_25_Acc_SIZE_32_18]


def MLP_CNN_Compared():
    name = ['MLP_hidden_node_50', 'MLP_hidden_node_25',
            'CNN_hidden_node_50', 'CNN_hidden_node_25']

    # Accuracy Compared
    plt.plot(Image_size, MLP_hidden_node_50_Acc)
    plt.plot(Image_size, MLP_hidden_node_25_Acc)
    plt.plot(Image_size, CNN_hidden_node_50_Acc)
    plt.plot(Image_size, CNN_hidden_node_25_Acc)
    plt.title('MLP_CNN_Compared_Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Image_size')
    plt.legend(name, loc='upper left')
    plt.show()

    # Loss Compared
    plt.plot(Image_size, MLP_hidden_node_50_Loss)
    plt.plot(Image_size, MLP_hidden_node_25_Loss)
    plt.plot(Image_size, CNN_hidden_node_50_Loss)
    plt.plot(Image_size, CNN_hidden_node_25_Loss)
    plt.title('MLP_CNN_Compared_Loss')
    plt.ylabel('Loss')
    plt.xlabel('Image_size')
    plt.legend(name, loc='upper left')
    plt.show()


def MLP_Compared():
    name = ['A', 'B']

    # MLP Compared
    fig, ax = plt.subplots(2, 1, constrained_layout=True)

    ax[0].plot(Image_size, MLP_hidden_node_50_Acc)
    ax[0].plot(Image_size, MLP_hidden_node_25_Acc)
    ax[0].set_title("MLP_compared_Accuracy")
    ax[0].set_xlabel("Image_size")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(name, loc='upper left')

    ax[1].plot(Image_size, MLP_hidden_node_50_Loss)
    ax[1].plot(Image_size, MLP_hidden_node_25_Loss)
    ax[1].set_title("MLP_compared_Loss")
    ax[1].set_xlabel("Image_size")
    ax[1].set_ylabel("Loss")
    ax[1].legend(name, loc='upper left')

    plt.show()


def CNN_Compared():
    name = ['A', 'B']

    # CNN Compared
    fig, ax = plt.subplots(2, 1, constrained_layout=True)

    ax[0].plot(Image_size, CNN_hidden_node_50_Acc)
    ax[0].plot(Image_size, CNN_hidden_node_25_Acc)
    ax[0].set_title("CNN_compared_Accuracy")
    ax[0].set_xlabel("Image_size")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(name, loc='upper left')

    ax[1].plot(Image_size, CNN_hidden_node_50_Loss)
    ax[1].plot(Image_size, CNN_hidden_node_25_Loss)
    ax[1].set_title("CNN_compared_Loss")
    ax[1].set_xlabel("Image_size")
    ax[1].set_ylabel("Loss")
    ax[1].legend(name, loc='upper left')

    plt.show()


def Crop_Compared():
    name = ['MLP B', 'Crop A(MLP)', 'CNN B', 'Crop B(CNN)']
    index = np.arange(len(MLP_CNN_Crop_Compared))
    colors = ['blue', 'red', 'blue', 'red']

    # MLP, CNN, Crop Model Compared
    plt.bar(index, MLP_CNN_Crop_Compared, color=colors, width=0.5)
    plt.title('MLP_CNN_Crop_Compared')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(98, 100)
    plt.xticks(index, name)

    plt.show()


# MLP_Compared()
# CNN_Compared()
# MLP_CNN_Compared()
Crop_Compared()
