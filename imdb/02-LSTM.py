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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np
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


# ==========================================================
#
# 資料預處理
#
# ==========================================================
print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=400)
X_test = sequence.pad_sequences(X_test, maxlen=400)


# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')

model = Sequential()
model.add(Embedding(input_dim=max_features,
                    output_dim=50,
                    input_length=400))
model.add(Dropout(0.35))

model.add(LSTM(32))

model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())

# ==========================================================
#
# 訓練模型
#
# ==========================================================
print('Training Model...')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=100,
          validation_split=0.2,
          epochs=2,
          verbose=2)

# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
print('Evaluate Model...')

scores = model.evaluate(X_test, y_test, verbose=1)
print()
print(scores[1])

# ==========================================================
#
# 進行預測
#
# ==========================================================
print('Predict...')

