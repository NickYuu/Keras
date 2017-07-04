#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/7/4
"""
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ==========================================================
#
# 環境
#
# ==========================================================

np.random.seed(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==========================================================
#
# Helper
#
# ==========================================================


def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    plt.figure()
    if num > 25:
        num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


def show_predicted_probability(predicted__probability, i):
    print('label:', label_dict[y_test[i][0]],
          'predict:', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + ' Probability:%1.9f' % (predicted__probability[i][j]))


# ==========================================================
#
# 資料預處理
#
# ==========================================================
print('Loading data...')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
label_dict = {0: "airplane",
              1: "automobile",
              2: "bird",
              3: "cat",
              4: "deer",
              5: "dog",
              6: "frog",
              7: "horse",
              8: "ship",
              9: "truck"}

# X normalize
X_train_normalize = X_train.astype('float32') / 255
X_test_normalize = X_test.astype('float32') / 255

# y OneHot Encoding
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')


model = Sequential()

# 卷積層1 與 池化層1
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(32, 32, 3),
                 activation='relu'))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(32, 32, 3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# 卷積層2 與 池化層2
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


# 卷積層3 與 池化層3
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


# 建立神經網路(平坦層、隱藏層、輸出層)
model.add(Flatten())
model.add(Dropout(0.3))

model.add(Dense(units=2048,
                activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=1024,
                activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=10,
                activation='softmax'))



# model = Sequential()
# model.add(ZeroPadding2D((1, 1), input_shape=(32, 32, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))

model.summary()

# ==========================================================
#
# 訓練模型
#
# ==========================================================
print('Training Model...')

# noinspection PyBroadException
try:
    model.load_weights("SaveModel/VGGModel.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(X_train_normalize,
                          y_trainOneHot,
                          batch_size=32,
                          epochs=1,
                          verbose=2,
                          validation_split=0.2)

model.save_weights("SaveModel/VGGModel.h5")
print("Saved model to disk")


# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
print('Evaluate Model...')

# show_train_history('acc', 'val_acc')
# show_train_history('loss', 'val_loss')

scores = model.evaluate(X_test_normalize, y_testOneHot, verbose=0)
print()
print(scores[1])


# ==========================================================
#
# 進行預測
#
# ==========================================================
print('Predict...')

prediction = model.predict_classes(X_test_normalize)
plot_images_labels_prediction(X_test, y_test, prediction, 0)

df = pd.DataFrame({'label': y_test.reshape(-1), 'predict': prediction})
index = df[df.label != df.predict].index
plot_images_labels_prediction(X_test[index], y_test[index], prediction[index], 0)

Predicted_Probability = model.predict(X_test_normalize)
show_predicted_probability(Predicted_Probability, 0)




