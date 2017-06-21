#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/6/21
"""
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Flatten, Embedding, GlobalAveragePooling1D
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
max_features = 5000

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


def display_test_Sentiment(i):
    SentimentDict = {1: '正面的', 0: '負面的'}
    print(X_test[i])
    print('標籤label:', SentimentDict[y_test[i]],
          '預測結果:', SentimentDict[predict[i, 0]])

# ==========================================================
#
# 資料預處理
#
# ==========================================================
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=400)
X_test = sequence.pad_sequences(X_test, maxlen=400)

# ==========================================================
#
# 建立模型
#
# ==========================================================

model = Sequential()
model.add(Embedding(input_dim=max_features,
                    output_dim=50,
                    input_length=400))
model.add(Dropout(0.35))

model.add(SimpleRNN(units=16))

model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())

# ==========================================================
#
# 訓練模型
#
# ==========================================================
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(X_train, y_train,
                          batch_size=100,
                          epochs=2,
                          verbose=2,
                          validation_data=(X_train, y_train))

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
scores = model.evaluate(X_test, y_test, verbose=1)
print(scores[1])


# ==========================================================
#
# 進行預測
#
# ==========================================================

predict = model.predict_classes(X_test)

df = pd.DataFrame({'label': y_test, 'predict': predict.reshape(-1)})
index = df[df.label != df.predict].index
print('\n')
display_test_Sentiment(index[6])
