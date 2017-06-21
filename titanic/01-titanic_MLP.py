import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_train_history(train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def preprocess_data(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df, columns=["embarked"])

    ndarray = x_OneHot_df.values
    features = ndarray[:, 1:]
    label = ndarray[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(features)

    return scaledFeatures, label


# 讀取資料
all_df = pd.read_csv('data/titanic3.csv')

# 資料預處理
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]

msk = np.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

train_features, train_label = preprocess_data(train_df)
test_features, test_label = preprocess_data(test_df)

# 建立模型
model = Sequential()

model.add(Dense(units=40,
                activation='relu',
                kernel_initializer='uniform',
                input_dim=9))
# model.add(Dropout(0.25))

model.add(Dense(units=30,
                activation='relu',
                kernel_initializer='uniform'))
# model.add(Dropout(0.25))
#
# model.add(Dense(units=10,
#                 activation='relu',
#                 kernel_initializer='uniform'))
# model.add(Dropout(0.25))

model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

print(model.summary())

# 訓練模型

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_history = model.fit(train_features,
                          train_label,
                          validation_split=0.2,
                          verbose=2,
                          batch_size=30,
                          epochs=30)

# 評估模型準確率
show_train_history('acc', 'val_acc')
show_train_history('loss', 'val_loss')

scores = model.evaluate(test_features, test_label)
print()
print(scores[1])

# 進行預測
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])
JR_df = pd.DataFrame([list(Jack), list(Rose)],
                     columns=['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])
all_df = pd.concat([all_df, JR_df])

all_Features, Label = preprocess_data(all_df)
all_probability = model.predict(all_Features)
data = all_df
data.insert(len(all_df.columns), 'probability', all_probability)
print(data[-2:])
print('生存率很高卻沒生存')
print(data[(data.survived == 0) & (data.probability > 0.9)])

