from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
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
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# 取得數據
(X_train_img, y_train_lab), (X_test_img, y_test_lab) = mnist.load_data()

# 歸一化
X_train = X_train_img.reshape(60000, 28 * 28).astype('float32') / 255
X_test = X_test_img.reshape(10000, 28 * 28).astype('float32') / 255

# one hot encoding
y_train = np_utils.to_categorical(y_train_lab)
y_test = np_utils.to_categorical(y_test_lab)

# 建立模型
model = Sequential()

# 將「輸入層」與「隱藏層」加入模型
model.add(Dense(units=1000,
                input_dim=28 * 28,
                activation='relu',
                kernel_initializer='normal'))
model.add(Dropout(0.5))


# 將「輸出層」加入模型
model.add(Dense(10,
                kernel_initializer='normal',
                activation='softmax'))

# 查看神經網路層
print(model.summary())

# 訓練模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(X_train,
                          y_train,
                          batch_size=200,
                          epochs=10,
                          validation_split=0.2,
                          verbose=2)

# 以圖形顯示訓練過程
show_train_history(train_history, 'acc', 'val_acc')

# 評估模型準確率
scores = model.evaluate(X_test, y_test)
print()
print(scores[1])

# 進行預測
prediction = model.predict_classes(X_test)

df = pd.DataFrame({'label': y_test_lab, 'predict': prediction})
error_prediction = df[(df.label == 5) & (df.label != df.predict)]
error_index = error_prediction.index
print(error_prediction)

plot_images_labels_prediction(X_test_img[error_index], y_test_lab[error_index], prediction[error_index], idx=1)

# confusion matrix 混淆矩陣
crosstab = pd.crosstab(y_test_lab, prediction, rownames=['y_test'], colnames=['predict'])
print()
print(crosstab)

