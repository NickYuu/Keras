import pandas as pd


data = pd.read_excel('titanic3.xls')
print(data.head())
data.to_csv('titanic3.csv')
