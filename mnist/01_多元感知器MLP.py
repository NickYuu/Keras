from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(10)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 顯示多張圖片
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
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


# 資料預處理
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()

print('X train', X_train_image.shape)
print('y train', X_test_image.shape)

X_train = X_train_image.reshape(60000, 28 * 28).astype('float32') / 255
X_test = X_test_image.reshape(10000, 28 * 28).astype('float32') / 255

y_train = np_utils.to_categorical(y_train_label)
y_test = np_utils.to_categorical(y_test_label)

# 建立模型
model = Sequential()
model.add(Dense(units=256,
                input_dim=28 * 28,
                kernel_initializer='normal',
                activation='relu'
                ))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'
                ))

print(model.summary())

# 訓練模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(x=X_train,
                          y=y_train,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=2)

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')


# 評估模型準確率
scores = model.evaluate(X_test, y_test)
print()
print(scores)


# 進行預測
prediction = model.predict_classes(X_test)
print()
print(prediction)

plot_images_labels_prediction(X_test_image, y_test_label, prediction, 340)

# confusion matrix 混淆矩陣
crosstab = pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
print()
print(crosstab)


df = pd.DataFrame({'label': y_test_label, 'predict': prediction})

error_prediction = df[df.label != df.predict]

error_index = error_prediction.index

# error_prediction = error_prediction.reindex(range(len(error_prediction)))
# print(error_prediction)

plot_images_labels_prediction(X_test_image[error_index], y_test_label[error_index], prediction[error_index], idx=100)
