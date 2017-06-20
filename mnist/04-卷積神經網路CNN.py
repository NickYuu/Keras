from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import os
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Convolutional Neural Network


def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_images_labels_prediction(images, labels, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" + str(labels[idx]) +
                     ",predict=" + str(prediction[idx])
                     , fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


"""
資料預處理
"""
# 取得資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 將Ｘ Reshape 為 （-1, 28, 28, 1）(數量, 長, 寬, 高), 並且歸一化
X_train4D = X_train.reshape(-1, 28, 28, 1) / 255
X_test4D = X_test.reshape(-1, 28, 28, 1) / 255

# 將 y 做 OneHot encoding
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

"""
建立模型
"""
# 建立模型
model = Sequential()

# 建立卷積層1 Convolutional Layer
model.add(Conv2D(filters=16,  # 建立16個濾鏡
                 kernel_size=(5, 5),  # 每個濾鏡 5 * 5 的大小
                 padding='same',  # 產生的卷積影像大小不變
                 input_shape=(28, 28, 1),  # 輸入的維度
                 activation='relu'))  # 設定ReLU激活函數

# 建立池化層1 Pooling Layer > 將原本 16 個 28*28 的影像轉換為 16 個 14*14
model.add(MaxPool2D(pool_size=(2, 2)))

# 建立卷積層2 Convolutional Layer > 將原本16個影像轉為 36個
model.add(Conv2D(filters=36,  # 建立16個濾鏡
                 kernel_size=(5, 5),  # 每個濾鏡 5 * 5 的大小
                 padding='same',  # 產生的卷積影像大小不變
                 activation='relu'))  # 設定ReLU激活函數

# 建立池化層2 Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# 加入 Dropout 避免 overfitting
model.add(Dropout(0.25))

# 建立平坦層 將 36 個 7*7 的影像 對應到 36*7*7 個神經元
model.add(Flatten())

# 建立隱藏層
model.add(Dense(units=128,
                activation='relu'))

# 加入 Dropout 避免 overfitting
model.add(Dropout(0.5))

# 建立輸出層
model.add(Dense(units=10,
                activation='softmax'))

# 查看建立的模型
print(model.summary())

"""
訓練模型
"""
# 定義訓練方式
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 開始訓練
train_history = model.fit(X_train4D, y_trainOneHot,
                          batch_size=300,
                          epochs=10,
                          validation_split=0.2,
                          verbose=2)

"""
評估模型準確率
"""

# 圖形化準確度及誤差
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# 評估模型
scores = model.evaluate(X_test4D, y_testOneHot)
print()
print(scores[1])

"""
進行預測
"""

prediction = model.predict_classes(X_test4D)

df = pd.DataFrame({'label': y_test,
                   'predict': prediction})

error_prediction = df[df.label != df.predict]
index = error_prediction.index

plot_images_labels_prediction(X_test[index], y_test[index], prediction[index], idx=0)


matrix = pd.crosstab(df.label, df.predict)
print()
print(matrix)
