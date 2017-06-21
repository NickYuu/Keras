#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/6/21
"""
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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


def rm_tags(text):
    return re.compile(r'<[^>]+>').sub('', text)


def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


def display_test_Sentiment(i):
    SentimentDict = {1: '正面的', 0: '負面的'}
    print(test_text[i])
    print('標籤label:', SentimentDict[y_test[i]],
          '預測結果:', SentimentDict[predict[i, 0]])


# ==========================================================
#
# 資料預處理
#
# ==========================================================

# 取得資料
y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

# 先讀取所有文章建立字典，限制字典的數量為nb_words=2000
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

# 將每一篇文章的文字轉換一連串的數字 只有在字典中的文字會轉換為數字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

# 讓轉換後的數字長度相同
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)

# ==========================================================
#
# 建立模型
#
# ==========================================================

model = Sequential()

# Embedding層
model.add(Embedding(
    input_dim=2000,
    output_dim=32,
    input_length=100
))
model.add(Dropout(0.2))

# Flatten層
model.add(Flatten())

# 隱藏層
model.add(Dense(
    units=256,
    activation='relu'
))
model.add(Dropout(0.35))

# 輸出層
model.add(Dense(
    units=1,
    activation='sigmoid'
))

print(model.summary())

# ==========================================================
#
# 訓練模型
#
# ==========================================================

try:
    model.load_weights("MLPModel.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(x_train,
                          y_train,
                          batch_size=100,
                          epochs=1,
                          verbose=2,
                          validation_split=0.2)

model.save_weights("MLPModel.h5")
print("模型保存成功")

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores[1])

# ==========================================================
#
# 進行預測
#
# ==========================================================

predict = model.predict_classes(x_test)

df = pd.DataFrame({'label': y_test, 'predict': predict.reshape(-1)})
index = df[df.label != df.predict].index
print('\n')
display_test_Sentiment(index[6])

