import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/titanic3.csv')


# 取用這些欄位
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp',
        'parch', 'fare', 'embarked']
data = data[cols]

# 缺失值處理
# print(data.isnull().sum())

df = data.drop('name', axis=1)

print(df.embarked.value_counts())
df.age = df.age.fillna(df.age.mean())
df.fare = df.fare.fillna(df.fare.mean())
df.embarked = df.embarked.fillna('S')
df.sex = df.sex.map({'female': 0, 'male': 1}).astype(int)

dfOneHot = pd.get_dummies(df, columns=['embarked'])


