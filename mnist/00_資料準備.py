import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist


# 顯示圖片
def plot_image(image):
    # fig = plt.gcf()
    # fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


# 顯示多張圖片
def plot_images_labels_prediction(images, labels,
                                  prediction, idx, num=10):
    # fig = plt.gcf()
    # fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# 載入數據
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('train data=', len(X_train))
print(' test data=', len(X_test))
print('image:', X_train.shape)
print('label:', y_train.shape)

# 測試印出第一張圖
# plot_image(X_train[0])

X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

print('x_train:', X_train.shape)
print('x_test:', X_test.shape)

x_Train_normalize = X_train / 255
x_Test_normalize = X_test / 255


# one hot encode
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

