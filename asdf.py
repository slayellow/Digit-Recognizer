import tensorflow as tf
import pandas as pd
data = pd.read_csv('./all/test.csv', sep=",", dtype='unicode')   # Data Load

print(data.columns)
print(data.head())
asdf
train_label = data["label"] # Train Data의 Label 값 저장
train_data = data[data.columns[1:]] # Train Data의 값들 저장