# -*- coding: utf-8 -*-
"""
随机切分数据集脚本
"""

import numpy as np
import pandas as pd

file_path = 'C:/Users/chenzhiwei/Desktop/automl/data/'

df = pd.read_csv(file_path + 'creditcard.csv')

ratio = 0.25

df_index = df.index.values
np.random.shuffle(df_index)
X_train_index = df_index[: int(df.shape[0] * 0.75)]
X_test_index = df_index[int(df.shape[0] * 0.75):]
train_set = df.iloc[X_train_index, :]
test_set = df.iloc[X_test_index, :]

train_set['Time'] = train_set['Time'].astype(int)
test_set['Time'] = test_set['Time'].astype(int)

train_set.to_csv(file_path + 'train_set_new.csv', index=False)
test_set.to_csv(file_path + 'test_set_new.csv', index=False)

print(1)
